import itertools

import numpy as np
import pandas as pd

from functools import partial
from pathos.pools import ProcessPool
from numpy.random import SeedSequence

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV

from stratify import stratify_classes, stratify_groups_classes
from calc_scoring import get_confusion_mtx, calc_score

from OrdinalClassifPipeline import OrdinalClassifPipeline

import config


# - * - * - * - * - * - #
# core function: fitting and predicting, with cross-validated (inner) calibration
# - * - * - * - * - * - #
# separate for parallelization
def run_calibr_fit_predict(decisor, opts_cv, opts_score, opts_pipe, hyperparams, df_trval, df_test, random_seq):
    # collect problem info
    num_classes = config.NUM_CLASSES
    labels = config.LABELS
    cost_mtx = config.COST_MTX

    # collect info regarding inner (calibration) cross validation setup
    n_folds_in = opts_cv['n_folds_in']

    # create inner folds: train&valid -> train vs valid, stratified by severity outcome and hospital
    # l_df_train, l_df_valid = stratify_classes(df_trval, config.VAR_CLASSIF,
    #                                            n_folds=n_folds_in, shuffle=config.SHUFFLE, random_seq=random_seq)
    l_df_train, l_df_valid = stratify_groups_classes(df_trval, config.VAR_GROUP, config.VAR_CLASSIF,
                                                     n_folds=n_folds_in, shuffle=config.SHUFFLE, random_seq=random_seq)
    # prepare inner fold iterators
    cv_list = []
    for n in range(n_folds_in):
        df_train = l_df_train[n]
        df_valid = l_df_valid[n]

        df_merge_train = df_trval.reset_index().merge(df_train, how='inner').set_index('index')
        df_merge_valid = df_trval.reset_index().merge(df_valid, how='inner').set_index('index')

        l_idx_train = df_merge_train.index.tolist()
        l_idx_valid = df_merge_valid.index.tolist()

        cv_list.append((l_idx_train, l_idx_valid))

    # prepare data
    if config.EXCLUDE_HOSPITAL:
        X_trval = df_trval.drop(columns=config.VARS_STRATIF)
        X_test = df_test.drop(columns=config.VARS_STRATIF)
    else:
        X_trval = df_trval.drop(columns=[config.VAR_CLASSIF])
        X_test = df_test.drop(columns=[config.VAR_CLASSIF])
    y_trval = df_trval[config.VAR_CLASSIF]
    y_test = df_test[config.VAR_CLASSIF]

    # prepare pipe
    pipe = OrdinalClassifPipeline(num_classes=num_classes, labels=labels, cost_mtx=cost_mtx,
                                  decisor=decisor, scorer=opts_score, options=opts_pipe, hyperparams=hyperparams,
                                  random_seq=random_seq)

    # fit uncalibrated pipeline
    pipe_unclbr = clone(pipe)
    pipe_unclbr.fit(X_trval, y_trval)

    # fit calibrated pipeline
    pipe_calibr = CalibratedClassifierCV(base_estimator=pipe,
                                         method=config.CALIBR_METHOD, ensemble=config.CALIBR_ENSEMBLE,
                                         cv=cv_list)
    pipe_calibr.fit(X_trval, y_trval)

    # predict uncalibrated pipeline
    y_pred_unclbr = pipe_unclbr.predict(X_test)
    if isinstance(y_test, pd.Series):
        y_pred_unclbr = pd.Series(y_pred_unclbr)
    proba_pred_unclbr = pipe_unclbr.predict_proba(X_test)

    # predict calibrated pipeline
    y_pred_calibr = pipe_calibr.predict(X_test)
    if isinstance(y_test, pd.Series):
        y_pred_calibr = pd.Series(y_pred_calibr)
    proba_pred_calibr = pipe_calibr.predict_proba(X_test)

    return y_test, \
           y_pred_unclbr, proba_pred_unclbr, pipe_unclbr,\
           y_pred_calibr, proba_pred_calibr, pipe_calibr


# - * - * - * - * - * - #
# outer cross validation
# - * - * - * - * - * - #
def run_calibr_cv(df_data, decisor, opts_cv, opts_score, opts_pipe, hyperparams, random_seq):
    # collect problem info
    num_classes = config.NUM_CLASSES
    labels = config.LABELS
    cost_mtx = config.COST_MTX

    # collect info regarding target score
    scorer_name = opts_score['scorer_name']
    scorer_args = opts_score['scorer_args']

    # collect info regarding outer cross validation setup
    n_folds_out = opts_cv['n_folds_out']

    # for each outer fold, generate independent random sequences by spawning
    random_seq_cv = random_seq.spawn(n_folds_out)

    # create outer folds: all -> train&valid vs test, stratified by severity outcome and hospital
    # l_df_trval, l_df_test = stratify_classes(df_data, config.VAR_CLASSIF,
    #                                          n_folds=n_folds_out, shuffle=config.SHUFFLE, random_seq=random_seq)
    l_df_trval, l_df_test = stratify_groups_classes(df_data, config.VAR_GROUP, config.VAR_CLASSIF,
                                                    n_folds=n_folds_out, shuffle=config.SHUFFLE, random_seq=random_seq)

    # run parallel version
    if config.PARALLEL_CALIBR_CV:
        # initialize parallel computation
        pool_calibr = ProcessPool(nodes=config.NUM_NODES_CLASSIF_OUT_CV)
        # create partial function
        f_calibr = partial(run_calibr_fit_predict, decisor, opts_cv, opts_score, opts_pipe, hyperparams)

        l_results_calibr = pool_calibr.map(f_calibr, l_df_trval, l_df_test, random_seq_cv)

        # close parallel computation
        pool_calibr.close()
        pool_calibr.join()
        pool_calibr.clear()
        del pool_calibr

    # run sequential version
    else:
        l_results_calibr = []
        for df_trval, df_test, rdm_seq_cv in zip(l_df_trval, l_df_test, random_seq_cv):
            y_true_calibr, y_pred_unclbr, proba_pred_unclbr, pipe_unclbr, y_pred_calibr, proba_pred_calibr, pipe_calibr = \
                run_calibr_fit_predict(decisor, opts_cv, opts_score, opts_pipe, hyperparams, df_trval, df_test, rdm_seq_cv)
            results_calibr = [y_true_calibr, y_pred_unclbr, proba_pred_unclbr, pipe_unclbr, y_pred_calibr, proba_pred_calibr, pipe_calibr]
            l_results_calibr.append(results_calibr)

    # reorder results from the parallel output format
    l_results_calibr = list(map(list, zip(*l_results_calibr)))
    y_true, y_pred_unclbr, proba_pred_unclbr, pipe_unclbr, y_pred_calibr, proba_pred_calibr, pipe_calibr = l_results_calibr

    # aggregate results from all folds
    y_true_cv = [y_true.tolist() for y_true in y_true]
    y_pred_unclbr_cv = [y_pred.tolist() for y_pred in y_pred_unclbr]
    y_pred_calibr_cv = [y_pred.tolist() for y_pred in y_pred_calibr]
    y_true_cv = list(itertools.chain(*y_true_cv))
    y_pred_unclbr_cv = list(itertools.chain(*y_pred_unclbr_cv))
    y_pred_calibr_cv = list(itertools.chain(*y_pred_calibr_cv))
    y_true_cv = pd.Series(y_true_cv)
    y_pred_unclbr_cv = pd.Series(y_pred_unclbr_cv)
    y_pred_calibr_cv = pd.Series(y_pred_calibr_cv)

    proba_pred_unclbr_cv = np.concatenate(proba_pred_unclbr, axis=0)
    proba_pred_calibr_cv = np.concatenate(proba_pred_calibr, axis=0)

    # evaluate performance score across all folds
    score_unclbr_cv = calc_score(scorer_name, y_true_cv, y_pred_unclbr_cv, y_pred_unclbr_cv,
                                 labels=labels, cost_mtx=cost_mtx, extra_args=scorer_args)
    score_calibr_cv = calc_score(scorer_name, y_true_cv, y_pred_calibr_cv, proba_pred_calibr_cv,
                                 labels=labels, cost_mtx=cost_mtx, extra_args=scorer_args)

    # calculate confusion matrix
    confus_mtx_unclbr_cv = get_confusion_mtx(y_true_cv, y_pred_unclbr_cv, labels)
    confus_mtx_calibr_cv = get_confusion_mtx(y_true_cv, y_pred_calibr_cv, labels)

    if config.VERBOSE >= 1:
        print('Calibration CV | Uncalibr. score [{}]: {:.5f}'.format(scorer_name, score_unclbr_cv), flush=True)
        confus_df_unclbr_cv = pd.DataFrame(confus_mtx_unclbr_cv, columns=config.TAGS_PRED, index=config.TAGS_TRUE)
        print('Calibration CV | Uncalibr. confusion matrix:\n{}'.format(confus_df_unclbr_cv), flush=True)

        print('Calibration CV | Calibrat. score [{}]: {:.5f}'.format(scorer_name, score_calibr_cv), flush=True)
        confus_df_calibr_cv = pd.DataFrame(confus_mtx_calibr_cv, columns=config.TAGS_PRED, index=config.TAGS_TRUE)
        print('Calibration CV | Calibrat. confusion matrix:\n{}'.format(confus_df_calibr_cv), flush=True)

    return y_true_cv, \
           y_pred_unclbr_cv, proba_pred_unclbr_cv, score_unclbr_cv, confus_mtx_unclbr_cv, \
           y_pred_calibr_cv, proba_pred_calibr_cv, score_calibr_cv, confus_mtx_calibr_cv


# - * - * - * - * - * - #
# repeated nested cross validation
# - * - * - * - * - * - #
# to assess variability of performance scores across iteration
def run_calibr_repeat_cv(df_data, decisor, opts_cv, opts_score, opts_pipe, opts_hyperparams, seed=None):
    n_iters = opts_cv['n_iters']

    # for each repetition, generate independent random sequences by spawning
    seed_seq = SeedSequence(seed)
    random_seq_repeat = seed_seq.spawn(n_iters)

    # run parallel version
    if config.PARALLEL_CALIBR_REPEAT:
        # initialize parallel computation
        pool_repeat = ProcessPool(nodes=config.NUM_NODES_CLASSIF_REPEAT)
        # create partial function
        f_calibr_cv = partial(run_calibr_cv, df_data, decisor, opts_cv, opts_score, opts_pipe, opts_hyperparams)

        l_results_calibr_repeat = pool_repeat.map(f_calibr_cv, random_seq_repeat)

        # close parallel computation
        pool_repeat.close()
        pool_repeat.join()
        pool_repeat.clear()
        del pool_repeat

    # run sequential version
    else:
        l_results_calibr_repeat = []
        for random_seq in random_seq_repeat:
            y_true_cv, \
            y_pred_unclbr_cv, proba_pred_unclbr_cv, score_unclbr_cv, confus_mtx_unclbr_cv, \
            y_pred_calibr_cv, proba_pred_calibr_cv, score_calibr_cv, confus_mtx_calibr_cv = \
                run_calibr_cv(df_data, decisor, opts_cv, opts_score, opts_pipe, opts_hyperparams, random_seq)
            results_calibr_cv = [y_true_cv,
                                 y_pred_unclbr_cv, proba_pred_unclbr_cv, score_unclbr_cv, confus_mtx_unclbr_cv,
                                 y_pred_calibr_cv, proba_pred_calibr_cv, score_calibr_cv, confus_mtx_calibr_cv]
            l_results_calibr_repeat.append(results_calibr_cv)

    # reorder results from the parallel output format
    l_results_calibr_repeat = list(map(list, zip(*l_results_calibr_repeat)))
    y_true_repeat, y_pred_unclbr_repeat, proba_pred_unclbr_repeat, score_unclbr_repeat, confus_mtx_unclbr_repeat, \
            y_pred_calibr_repeat, proba_pred_calibr_repeat, score_calibr_repeat, confus_mtx_calibr_repeat = \
        l_results_calibr_repeat

    return y_true_repeat, y_pred_unclbr_repeat, proba_pred_unclbr_repeat, score_unclbr_repeat, confus_mtx_unclbr_repeat, \
            y_pred_calibr_repeat, proba_pred_calibr_repeat, score_calibr_repeat, confus_mtx_calibr_repeat
