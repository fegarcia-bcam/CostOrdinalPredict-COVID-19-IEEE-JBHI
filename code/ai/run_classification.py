import itertools

import numpy as np
import pandas as pd

from functools import partial
from pathos.pools import ProcessPool
from numpy.random import SeedSequence

from stratify import stratify_classes, stratify_groups_classes
from utils_hyperparams import hyperparams_sample, hyperparams_unflatten
from calc_scoring import get_confusion_mtx, calc_score

from OrdinalClassifPipeline import OrdinalClassifPipeline

import config


# - * - * - * - * - * - #
# core function: fitting and predicting
# - * - * - * - * - * - #
# separate for parallelization
def run_classif_fit_predict(is_inner_cv, decisor, opts_score, opts_pipe, hyperparams, df_train, df_test, random_seq):
    # collect problem info
    num_classes = config.NUM_CLASSES
    labels = config.LABELS
    cost_mtx = config.COST_MTX

    # collect info regarding target score
    scorer_name = opts_score['scorer_name']
    scorer_args = opts_score['scorer_args']

    # prepare data
    if config.EXCLUDE_HOSPITAL:
        X_train = df_train.drop(columns=config.VARS_STRATIF)
        X_test = df_test.drop(columns=config.VARS_STRATIF)
    else:
        X_train = df_train.drop(columns=[config.VAR_CLASSIF])
        X_test = df_test.drop(columns=[config.VAR_CLASSIF])
    y_train = df_train[config.VAR_CLASSIF]
    y_test = df_test[config.VAR_CLASSIF]

    # reformat hyperparams
    hyperparams = hyperparams_unflatten(hyperparams, is_classif=True)

    # prepare pipe
    pipe = OrdinalClassifPipeline(num_classes=num_classes, labels=labels, cost_mtx=cost_mtx,
                                  decisor=decisor, scorer=opts_score, options=opts_pipe, hyperparams=hyperparams,
                                  random_seq=random_seq)

    # fit model
    pipe.fit(X_train, y_train)

    # predict with fitted model
    y_pred, proba_pred = pipe.predict_pipe(X_test)

    # evaluate performance score in current fold
    score = calc_score(scorer_name, y_test, y_pred, proba_pred,
                       labels=labels, cost_mtx=cost_mtx, extra_args=scorer_args)

    str_cv, verbose_thr = ('Inner CV', 3) if is_inner_cv else ('Outer CV', 1)
    if config.VERBOSE >= verbose_thr:
        print('{} | Fold score [{}]: {:.5f}'.format(str_cv, scorer_name, score), flush=True)

    return y_test, y_pred, proba_pred, pipe, score


# - * - * - * - * - * - #
# inner cross validation
# - * - * - * - * - * - #
def run_classif_in_cv(decisor, opts_cv, opts_score, opts_pipe, df_trval, hyperparams, random_seq):
    # collect problem info
    num_classes = config.NUM_CLASSES
    labels = config.LABELS
    cost_mtx = config.COST_MTX

    # collect info regarding inner cross validation setup
    n_folds_in = opts_cv['n_folds_in']

    # collect info regarding target score
    scorer_name = opts_score['scorer_name']
    scorer_args = opts_score['scorer_args']

    # for each inner fold, generate independent random sequences by spawning
    random_seq_in = random_seq.spawn(n_folds_in)

    if config.VERBOSE >= 3:
        print('Hyperparams | Curr setting:\n{}'.format(hyperparams), flush=True)

    # create inner folds: train&valid -> train vs valid, stratified by severity outcome and hospital
    # l_df_train, l_df_valid = stratify_classes(df_trval, config.VAR_CLASSIF,
    #                                            n_folds=n_folds_in, shuffle=config.SHUFFLE, random_seq=random_seq)
    l_df_train, l_df_valid = stratify_groups_classes(df_trval, config.VAR_GROUP, config.VAR_CLASSIF,
                                                     n_folds=n_folds_in, shuffle=config.SHUFFLE, random_seq=random_seq)

    # run parallel version
    if config.PARALLEL_CLASSIF_IN_CV:
        # initialize parallel computation
        pool_in_cv = ProcessPool(nodes=config.NUM_NODES_CLASSIF_IN_CV)
        # create partial function
        f_fit_pred = partial(run_classif_fit_predict, True, decisor, opts_score, opts_pipe, hyperparams)

        l_results_in_cv = pool_in_cv.map(f_fit_pred, l_df_train, l_df_valid, random_seq_in)

        # close parallel computation
        pool_in_cv.close()
        pool_in_cv.join()
        pool_in_cv.clear()
        del pool_in_cv

    # run sequential version
    else:
        l_results_in_cv = []
        for df_train, df_valid, rdm_seq_in in zip(l_df_train, l_df_valid, random_seq_in):
            y_true_in, y_pred_in, proba_pred_in, pipes_in, scores_in = \
                run_classif_fit_predict(True, decisor, opts_score, opts_pipe, hyperparams, df_train, df_valid, rdm_seq_in)
            results_in_cv = [y_true_in, y_pred_in, proba_pred_in, pipes_in, scores_in]
            l_results_in_cv.append(results_in_cv)

    # reorder results from the parallel output format
    l_results_in_cv = list(map(list, zip(*l_results_in_cv)))
    y_true_in_cv, y_pred_in_cv, proba_pred_in_cv, pipes_in_cv, scores_in_cv = l_results_in_cv

    # aggregate results from all inner folds
    y_true_in_cv = [y_true_in.tolist() for y_true_in in y_true_in_cv]
    y_pred_in_cv = [y_pred_in.tolist() for y_pred_in in y_pred_in_cv]
    y_true_in_cv = list(itertools.chain(*y_true_in_cv))
    y_pred_in_cv = list(itertools.chain(*y_pred_in_cv))
    y_true_in_cv = pd.Series(y_true_in_cv)
    y_pred_in_cv = pd.Series(y_pred_in_cv)

    proba_pred_in_cv = np.concatenate(proba_pred_in_cv, axis=0)

    # evaluate performance score across all inner folds
    score_in_cv = calc_score(scorer_name, y_true_in_cv, y_pred_in_cv, proba_pred_in_cv,
                             labels=labels, cost_mtx=cost_mtx, extra_args=scorer_args)
    if config.VERBOSE >= 2:
        print('Inner CV | Aggreg. score [{}]: {:.5f}'.format(scorer_name, score_in_cv), flush=True)

    return score_in_cv


# - * - * - * - * - * - #
# hyperparameter optimization
# - * - * - * - * - * - #
def run_classif_hyperparam_optim(decisor, opts_cv, opts_score, opts_pipe, opts_hyperparams, df_trval, random_seq):
    # prepare hyperparameter search
    hyperparam_search = opts_hyperparams['search']
    hyperparam_iters = opts_hyperparams['iters']
    hyperparam_set = opts_hyperparams['set']

    # first generate a sample collection of hyperparameters to search, either from grid or random distribution
    l_hyperparams = hyperparams_sample(hyperparam_set, hyperparam_search, hyperparam_iters, random_seq=random_seq)
    if len(l_hyperparams) == 1:
        idx_best = 0
        hyperparams_best = l_hyperparams[idx_best]
        if config.VERBOSE >= 1:
            print('Hyperparams | Unique setting:\n{}'.format(hyperparams_best), flush=True)
        return hyperparams_best

    # spawn a collection of random sequences of the same size of the hyperparameter search space
    random_seq_hyp = random_seq.spawn(len(l_hyperparams))

    # run parallel version
    if config.PARALLEL_CLASSIF_HYPERPARAM_OPT:
        # initialize parallel computation
        pool_hyp_opt = ProcessPool(nodes=config.NUM_NODES_CLASSIF_HYPERPARAM_OPT)
        # create partial function
        f_in_cv = partial(run_classif_in_cv, decisor, opts_cv, opts_score, opts_pipe, df_trval)

        l_scores_hyp = pool_hyp_opt.map(f_in_cv, l_hyperparams, random_seq_hyp)

        # close parallel computation
        pool_hyp_opt.close()
        pool_hyp_opt.join()
        pool_hyp_opt.clear()
        del pool_hyp_opt

    # run sequential version
    else:
        l_scores_hyp = []
        for hyperparams, rdm_seq_hyp in zip(l_hyperparams, random_seq_hyp):
            score_in_cv = run_classif_in_cv(decisor, opts_cv, opts_score, opts_pipe, df_trval, hyperparams, rdm_seq_hyp)
            l_scores_hyp.append(score_in_cv)

    # check which hyperparameter setting obtained best inner cross-val performance, i.e. maximum score
    l_scores_hyp = np.asarray(l_scores_hyp)
    idx_best = int(np.argmax(l_scores_hyp))  # scores convention is the higher the better
    hyperparams_best = l_hyperparams[idx_best]

    if config.VERBOSE >= 1:
        print('Hyperparams | Best setting:\n{}'.format(hyperparams_best), flush=True)

    return hyperparams_best


# - * - * - * - * - * - #
# core function: optimizing, fitting and predicting
# - * - * - * - * - * - #
# separate for parallelization
def run_classif_optim_fit_predict(decisor, opts_cv, opts_score, opts_pipe, opts_hyperparams, df_train, df_test, random_seq):
    # perform search for optimal hyperparameters
    hyperparams_best = run_classif_hyperparam_optim(decisor, opts_cv, opts_score, opts_pipe, opts_hyperparams, df_train, random_seq)
    # once found, fit and eval with those
    y_test, y_pred, proba_pred, pipe_best, score_best = \
        run_classif_fit_predict(False, decisor, opts_score, opts_pipe, hyperparams_best, df_train, df_test, random_seq)

    return y_test, y_pred, proba_pred, hyperparams_best, pipe_best, score_best


# - * - * - * - * - * - #
# outer cross validation
# - * - * - * - * - * - #
def run_classif_out_cv(df_data, decisor, opts_cv, opts_score, opts_pipe, opts_hyperparams, random_seq):
    # collect problem info
    num_classes = config.NUM_CLASSES
    labels = config.LABELS
    cost_mtx = config.COST_MTX

    # collect info regarding outer cross validation setup
    n_folds_out = opts_cv['n_folds_out']

    # collect info regarding target score
    scorer_name = opts_score['scorer_name']
    scorer_args = opts_score['scorer_args']

    if config.VERBOSE >= 1:
        print('Outer CV | Pipe options:\n{}'.format(opts_pipe), flush=True)

    # for each outer fold, generate independent random sequences by spawning
    random_seq_out = random_seq.spawn(n_folds_out)

    # create outer folds: all -> train&valid vs test, stratified by severity outcome and hospital
    # l_df_trval, l_df_test = stratify_classes(df_data, config.VAR_CLASSIF,
    #                                          n_folds=n_folds_out, shuffle=config.SHUFFLE, random_seq=random_seq)
    l_df_trval, l_df_test = stratify_groups_classes(df_data, config.VAR_GROUP, config.VAR_CLASSIF,
                                                    n_folds=n_folds_out, shuffle=config.SHUFFLE, random_seq=random_seq)

    # run parallel version
    if config.PARALLEL_CLASSIF_OUT_CV:
        # initialize parallel computation
        pool_out_cv = ProcessPool(nodes=config.NUM_NODES_CLASSIF_OUT_CV)
        # create partial function
        f_out_cv = partial(run_classif_optim_fit_predict, decisor, opts_cv, opts_score, opts_pipe, opts_hyperparams)

        l_results_out_cv = pool_out_cv.map(f_out_cv, l_df_trval, l_df_test, random_seq_out)

        # close parallel computation
        pool_out_cv.close()
        pool_out_cv.join()
        pool_out_cv.clear()
        del pool_out_cv

    # run sequential version
    else:
        l_results_out_cv = []
        for df_trval, df_test, rdm_seq_out in zip(l_df_trval, l_df_test, random_seq_out):
            y_true_out, y_pred_out, proba_pred_out, hyperparams_best, pipe_best, score_best = \
                run_classif_optim_fit_predict(decisor, opts_cv, opts_score, opts_pipe, opts_hyperparams, df_trval, df_test, rdm_seq_out)
            results_out_cv = [y_true_out, y_pred_out, proba_pred_out, hyperparams_best, pipe_best, score_best]
            l_results_out_cv.append(results_out_cv)

    # reorder results from the parallel output format
    l_results_out_cv = list(map(list, zip(*l_results_out_cv)))
    y_true_out_cv, y_pred_out_cv, proba_pred_out_cv, hyperparams_best_cv, pipe_best_cv, score_best_cv = l_results_out_cv

    # aggregate results from all inner folds
    y_true_out_cv = [y_true_out.tolist() for y_true_out in y_true_out_cv]
    y_pred_out_cv = [y_pred_out.tolist() for y_pred_out in y_pred_out_cv]
    y_true_out_cv = list(itertools.chain(*y_true_out_cv))
    y_pred_out_cv = list(itertools.chain(*y_pred_out_cv))
    y_true_out_cv = pd.Series(y_true_out_cv)
    y_pred_out_cv = pd.Series(y_pred_out_cv)

    proba_pred_out_cv = np.concatenate(proba_pred_out_cv, axis=0)

    # evaluate performance score across all outer folds
    score_out_cv = calc_score(scorer_name, y_true_out_cv, y_pred_out_cv, proba_pred_out_cv,
                              labels=labels, cost_mtx=cost_mtx, extra_args=scorer_args)

    # calculate confusion matrix
    confus_mtx_out_cv = get_confusion_mtx(y_true_out_cv, y_pred_out_cv, labels)

    # collect intermediate results from each outer cv fold, mainly to diagnose stability
    summary_out_cv = []
    for hyperparams_best, score_best in zip(hyperparams_best_cv, score_best_cv):
        summary_out = {'hyperparams_best': hyperparams_best, 'score_best': score_best}
        summary_out_cv.append(summary_out)

    if config.VERBOSE >= 1:
        print('Outer CV | Aggreg. score [{}]: {:.5f}'.format(scorer_name, score_out_cv), flush=True)

        confus_df_out_cv = pd.DataFrame(confus_mtx_out_cv, columns=config.TAGS_PRED, index=config.TAGS_TRUE)
        print('Outer CV | Confusion matrix:\n{}'.format(confus_df_out_cv), flush=True)

    return score_out_cv, confus_mtx_out_cv, y_true_out_cv, proba_pred_out_cv, summary_out_cv


# - * - * - * - * - * - #
# repeated nested cross validation
# - * - * - * - * - * - #
# to assess variability of performance scores across iteration
def run_classif_repeat_nest_cv(df_data, decisor, opts_cv, opts_score, opts_pipe, opts_hyperparams, seed=None):
    n_iters = opts_cv['n_iters']

    # for each repetition, generate independent random sequences by spawning
    seed_seq = SeedSequence(seed)
    random_seq_repeat = seed_seq.spawn(n_iters)

    # run parallel version
    if config.PARALLEL_CLASSIF_REPEAT:
        # initialize parallel computation
        pool_repeat = ProcessPool(nodes=config.NUM_NODES_CLASSIF_REPEAT)
        # create partial function
        f_out_cv = partial(run_classif_out_cv, df_data, decisor, opts_cv, opts_score, opts_pipe, opts_hyperparams)

        l_results_out_cv = pool_repeat.map(f_out_cv, random_seq_repeat)

        # close parallel computation
        pool_repeat.close()
        pool_repeat.join()
        pool_repeat.clear()
        del pool_repeat

    # run sequential version
    else:
        l_results_out_cv = []
        for random_seq in random_seq_repeat:
            score_out_cv, confus_mtx_out_cv, y_true_out_cv, proba_pred_out_cv, summary_out_cv = \
                run_classif_out_cv(df_data, decisor, opts_cv, opts_score, opts_pipe, opts_hyperparams, random_seq)
            results_out_cv = [score_out_cv, confus_mtx_out_cv,
                              y_true_out_cv, proba_pred_out_cv, summary_out_cv]
            l_results_out_cv.append(results_out_cv)

    # reorder results from the parallel output format
    l_results_out_cv = list(map(list, zip(*l_results_out_cv)))
    scores_repeat, confus_mtx_repeat, y_true_repeat, proba_pred_repeat, summary_repeat = l_results_out_cv

    return scores_repeat, confus_mtx_repeat, y_true_repeat, proba_pred_repeat, summary_repeat
