import warnings

import numpy as np
import pandas as pd

from functools import partial
from pathos.pools import ProcessPool
from numpy.random import SeedSequence

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV

from shap import kmeans, sample, KernelExplainer

from stratify import stratify_classes, stratify_groups_classes

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
                                         cv=cv_list, n_jobs=-1)
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
# SHAPLY explanation of features
# - * - * - * - * - * - #
def run_shap_expl_cv(decisor, opts_cv, opts_score, opts_pipe, hyperparams, df_trval, df_test, random_seq):
    # collect problem info
    feat_names_in = df_trval.columns.to_list()
    # feat_names_in.remove(config.VAR_GROUP)
    # feat_names_in.remove(config.VAR_CLASSIF)

    _, _, _, _, _, _, pipe_calibr = \
        run_calibr_fit_predict(decisor, opts_cv, opts_score, opts_pipe, hyperparams, df_trval, df_test, random_seq)

    f_calibr = lambda x: pipe_calibr.predict_proba(pd.DataFrame(x, columns=feat_names_in))

    if np.isfinite(config.K_SUMMARY_SHAP):
        # # rather than use the whole training set to estimate expected values, summarize with a set of weighted k-means
        # # may not work for categorical variables (?)
        # df_summary = kmeans(df_trval, k=config.K_SUMMARY_SHAP)
        # df_summary = pd.DataFrame(df_summary.data, columns=feat_names_in)

        # rather than use the whole training set to estimate expected values, summarize with a subsample
        df_summary = sample(df_trval, nsamples=config.K_SUMMARY_SHAP)
    else:
        df_summary = df_trval.copy()

    shap_explainer = KernelExplainer(model=f_calibr, data=df_summary,
                                     link='identity', algorithm='auto')

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')  # to avoid some non-applicable warnings
        shap_values = shap_explainer.shap_values(df_test, nsamples=config.NUM_EVALS_SHAP)

    shap_values = [shap_val.tolist() for shap_val in shap_values]
    return shap_values


# - * - * - * - * - * - #
# SHAPLY explanation of features
# - * - * - * - * - * - #
def run_shap_explanation(df_data, decisor, opts_cv, opts_score, opts_pipe, hyperparams, seed):
    # collect info regarding outer cross validation setup
    n_folds_out = opts_cv['n_folds_out']

    # generate random sequence
    seed_seq = SeedSequence(seed)
    random_seq_repeat = seed_seq.spawn(n_folds_out)

    # create outer folds: all -> train&valid vs test, stratified by severity outcome and hospital
    # l_df_trval, l_df_test = stratify_classes(df_data, config.VAR_CLASSIF,
    #                                          n_folds=n_folds_out, shuffle=config.SHUFFLE, random_seq=seed_seq)
    l_df_trval, l_df_test = stratify_groups_classes(df_data, config.VAR_GROUP, config.VAR_CLASSIF,
                                                    n_folds=n_folds_out, shuffle=config.SHUFFLE, random_seq=seed_seq)

    # run parallel version
    if config.PARALLEL_SHAP:
        # initialize parallel computation
        pool_repeat = ProcessPool(nodes=n_folds_out)
        # create partial function
        f_shap_cv = partial(run_shap_expl_cv, decisor, opts_cv, opts_score, opts_pipe, hyperparams)

        l_results_shap = pool_repeat.map(f_shap_cv, l_df_trval, l_df_test, random_seq_repeat)

        # close parallel computation
        pool_repeat.close()
        pool_repeat.join()
        pool_repeat.clear()
        del pool_repeat

    # run sequential version
    else:
        l_results_shap = []
        for df_trval, df_test, random_seq in zip(l_df_trval, l_df_test, random_seq_repeat):
            results_shap = run_shap_expl_cv(decisor, opts_cv, opts_score, opts_pipe, hyperparams, df_trval, df_test, random_seq)
            l_results_shap.append(results_shap)

    return l_results_shap
