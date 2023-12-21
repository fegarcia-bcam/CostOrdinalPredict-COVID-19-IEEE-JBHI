import os
import json

from datetime import datetime

import numpy as np
import pandas as pd

from numpy.random import SeedSequence

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV

from stratify import stratify_classes, stratify_groups_classes
from calc_scoring import get_confusion_mtx, calc_score

import config

# VERBOSE = 1
VERBOSE = 4  # maximum verbosity

# NUM_JOBS = None  # HistGradBoost is already parallel internally
NUM_JOBS = -1


def load_data():
    df_data = pd.read_csv(config.FILE_DATA_IN, header=0, sep=config.DELIMITER)
    if len(df_data.columns) == 1:
        raise SystemExit(config.MSSG_ERROR_DATA)

    df_data = df_data.drop(columns=config.VARS_EXTRA)

    return df_data


def choose_scenario():
    # choose nested cross validation scenario
    opts_cv = {'n_iters': 20,
               'n_folds_out': 10,
               'n_folds_in': 10}  # exhaustive performance assessment

    # choose performance score
    opts_score = {'scorer_name': 'custom',
                  'scorer_args': {'overall_average': 'harm',  # 'arithm', 'geom', 'harm'
                                  'list_scorer_names': ['cost_score',
                                                        'geo_mean_sens'],
                                  'list_extra_args': [{'average': 'micro', 'reference': 'base'},
                                                      {'average': 'multiclass',
                                                       'correction': 10.0 ** (-1 * config.NUM_CLASSES)}]
                                  }
                  }

    # choose hyperparams search space
    hyperparam_set = {'estimator__loss': ['log_loss'],
                      'estimator__learning_rate': [0.02, 0.05, 0.10, 0.20, 0.50],
                      'estimator__max_iter': [50, 100, 200],
                      'estimator__max_leaf_nodes': [None],
                      'estimator__max_depth': [10, 20, None]}  # HistGradBoost-classif

    # choose hyperparams search strategy
    opts_hyperparams = {'search': 'grid',
                        'set': hyperparam_set}

    return opts_cv, opts_score, opts_hyperparams


def build_baseline_ml_hgb():
    l_transf_encoder = []
    l_idx_categorical = []
    for idx_feat, feat_name in enumerate(config.VARS_IN):
        if feat_name in config.VARS_CATEGORICAL.keys():
            l_idx_categorical.append(idx_feat)
            l_categories = config.VARS_CATEGORICAL[feat_name]
            encoder_categ = OrdinalEncoder(categories=[l_categories],
                                           handle_unknown='use_encoded_value', unknown_value=np.nan)
            transf_onehot = (feat_name, encoder_categ, [feat_name])
            l_transf_encoder.append(transf_onehot)
        else:
            transf_pass = (feat_name, 'passthrough', [feat_name])
            l_transf_encoder.append(transf_pass)
    encoder = ColumnTransformer(l_transf_encoder)

    estimator = HistGradientBoostingClassifier(categorical_features=l_idx_categorical)

    base_ml_model = Pipeline([('encoder', encoder), ('estimator', estimator)])
    return base_ml_model


def run_baseline_ml_hgb(df_data, opts_cv, opts_score, opts_hyperparams, seed):
    base_ml_model = build_baseline_ml_hgb()

    n_iters = opts_cv['n_iters']
    n_folds_out = opts_cv['n_folds_out']
    n_folds_in = opts_cv['n_folds_in']

    # for each repetition, generate independent random sequences by spawning
    seed_seq = SeedSequence(seed)
    random_seq_repeat = seed_seq.spawn(n_iters)

    y_true_repeat, y_pred_unclbr_repeat, y_pred_calibr_repeat = [], [], []
    proba_pred_unclbr_repeat, proba_pred_calibr_repeat = [], []
    confus_mtx_unclbr_repeat, confus_mtx_calibr_repeat = [], []
    score_unclbr_repeat, score_calibr_repeat = [], []
    for idx_iter, random_seq in enumerate(random_seq_repeat):
        l_y_true, l_y_pred_unclbr , l_y_pred_calibr = [], [], []
        l_proba_pred_unclbr, l_proba_pred_calibr = [], []

        # prepare folds
        # l_df_train, l_df_test = stratify_classes(df_data, config.VAR_CLASSIF,
        #                                          n_folds=n_folds_out, shuffle=config.SHUFFLE, random_seq=random_seq)
        l_df_train, l_df_test = stratify_groups_classes(df_data, config.VAR_GROUP, config.VAR_CLASSIF,
                                                        n_folds=n_folds_out, shuffle=config.SHUFFLE, random_seq=random_seq)

        for idx_cv_out, (df_train, df_test) in enumerate(zip(l_df_train, l_df_test)):
            # prepare data
            if config.EXCLUDE_HOSPITAL:
                X_train = df_train.drop(columns=config.VARS_STRATIF)
                X_test = df_test.drop(columns=config.VARS_STRATIF)
            else:
                X_train = df_train.drop(columns=[config.VAR_CLASSIF])
                X_test = df_test.drop(columns=[config.VAR_CLASSIF])
            y_train = df_train[config.VAR_CLASSIF]
            y_test = df_test[config.VAR_CLASSIF]

            # prepare model uncalibrated
            model_unclbr = clone(base_ml_model)
            if opts_hyperparams['search'] != 'grid':
                raise RuntimeError
            model_unclbr = GridSearchCV(estimator=model_unclbr,
                                        param_grid=opts_hyperparams['set'],
                                        cv=StratifiedKFold(n_splits=n_folds_in),
                                        scoring='roc_auc_ovr', refit=True,
                                        verbose=VERBOSE, n_jobs=NUM_JOBS)
            # train model uncalibrated
            model_unclbr.fit(X_train, y_train)
            if VERBOSE:
                print('Model trained (GridSearchCV): Fold {}/{} [Iteration {}/{}]'.format(idx_cv_out + 1, n_folds_out, idx_iter + 1, n_iters))

            # prepare model calibrated
            model_calibr = clone(base_ml_model)
            model_calibr.set_params(**model_unclbr.best_params_)  # pick the best hyperparametes from the grid search
            model_calibr = CalibratedClassifierCV(base_estimator=model_calibr,
                                                  method='sigmoid',
                                                  cv=StratifiedKFold(n_splits=n_folds_in),
                                                  ensemble=True,
                                                  n_jobs=NUM_JOBS)

            # train model calibrated
            model_calibr.fit(X_train, y_train)
            if VERBOSE:
                print('Model trained (CalibrClassifCV): Fold {}/{} [Iteration {}/{}]'.format(idx_cv_out + 1, n_folds_out, idx_iter + 1, n_iters))

            # predict
            y_pred_test_unclbr = model_unclbr.predict(X_test)
            y_pred_test_calibr = model_calibr.predict(X_test)
            proba_pred_test_unclbr = model_unclbr.predict_proba(X_test)
            proba_pred_test_calibr = model_calibr.predict_proba(X_test)

            # concatenate intermediate results
            l_y_true += y_test.tolist()
            l_y_pred_unclbr += y_pred_test_unclbr.tolist()
            l_y_pred_calibr += y_pred_test_calibr.tolist()
            l_proba_pred_unclbr += proba_pred_test_unclbr.tolist()
            l_proba_pred_calibr += proba_pred_test_calibr.tolist()

        # compute confusion matrix and custom score
        confus_mtx_unclbr = get_confusion_mtx(l_y_true, l_y_pred_unclbr, labels=config.LABELS)
        confus_mtx_calibr = get_confusion_mtx(l_y_true, l_y_pred_calibr, labels=config.LABELS)
        score_unclbr = calc_score(y_true=l_y_true, y_pred=l_y_pred_unclbr, proba_pred=l_proba_pred_unclbr,
                                  labels=config.LABELS, cost_mtx=config.COST_MTX,
                                  scorer_name=opts_score['scorer_name'], extra_args=opts_score['scorer_args'])
        score_calibr = calc_score(y_true=l_y_true, y_pred=l_y_pred_calibr, proba_pred=l_proba_pred_calibr,
                                  labels=config.LABELS, cost_mtx=config.COST_MTX,
                                  scorer_name=opts_score['scorer_name'], extra_args=opts_score['scorer_args'])

        # append results from this iteration
        y_true_repeat.append(l_y_true)
        y_pred_unclbr_repeat.append(l_y_pred_unclbr)
        y_pred_calibr_repeat.append(l_y_pred_calibr)
        proba_pred_unclbr_repeat.append(l_proba_pred_unclbr)
        proba_pred_calibr_repeat.append(l_proba_pred_calibr)
        confus_mtx_unclbr_repeat.append(confus_mtx_unclbr)
        confus_mtx_calibr_repeat.append(confus_mtx_calibr)
        score_unclbr_repeat.append(score_unclbr)
        score_calibr_repeat.append(score_calibr)

        if VERBOSE:
            print('Iteration {}/{} finished'.format(idx_iter + 1, n_iters))

    return (y_true_repeat,
            y_pred_unclbr_repeat, proba_pred_unclbr_repeat, confus_mtx_unclbr_repeat, score_unclbr_repeat,
            y_pred_calibr_repeat, proba_pred_calibr_repeat, confus_mtx_calibr_repeat, score_calibr_repeat)


def save_results_baseline_ml_hgb(opts_cv, opts_score, opts_hyperparams, seed,
                                 y_true_repeat,
                                 y_pred_unclbr_repeat, proba_pred_unclbr_repeat, confus_mtx_unclbr_repeat, score_unclbr_repeat,
                                 y_pred_calibr_repeat, proba_pred_calibr_repeat, confus_mtx_calibr_repeat, score_calibr_repeat):
    # create dict containing all relevant info
    results = {'opts_cv': opts_cv, 'opts_score': opts_score, 'opts_hyperparams': opts_hyperparams, 'seed': seed,
               'y_true_repeat': y_true_repeat,
               'y_pred_unclbr_repeat': y_pred_unclbr_repeat,
               'y_pred_calibr_repeat': y_pred_calibr_repeat,
               'proba_pred_unclbr_repeat': proba_pred_unclbr_repeat,
               'proba_pred_calibr_repeat': proba_pred_calibr_repeat,
               'confus_mtx_unclbr_repeat': [cm.tolist() for cm in confus_mtx_unclbr_repeat],
               'confus_mtx_calibr_repeat': [cm.tolist() for cm in confus_mtx_calibr_repeat],
               'score_unclbr_repeat': score_unclbr_repeat,
               'score_calibr_repeat': score_calibr_repeat}

    # generate timestamp and a file in the desired path
    path = config.PATH_RESULTS_BASELINE
    if not os.path.exists(path):
        os.makedirs(path)
    timestamp = datetime.now().strftime(config.DATETIME_FORMAT)
    filename = config.FILE_RESULTS_BASELINE.format(timestamp)
    # print out timestamp
    print('Timestamp: {}'.format(timestamp), flush=True)

    # store to disk via json
    f_results = os.path.join(path, filename)
    with open(f_results, 'w') as f_json_results:
        json.dump(results, f_json_results)

    return f_results


if __name__ == '__main__':
    # import data file
    df_data = load_data()

    # choose scenario
    opts_cv, opts_score, opts_hyperparams = choose_scenario()

    # run repeated nested cross validation
    seed = config.SEED_CLASSIFY
    (y_true_repeat,
     y_pred_unclbr_repeat, proba_pred_unclbr_repeat, confus_mtx_unclbr_repeat, score_unclbr_repeat,
     y_pred_calibr_repeat, proba_pred_calibr_repeat, confus_mtx_calibr_repeat, score_calibr_repeat) = \
        run_baseline_ml_hgb(df_data, opts_cv, opts_score, opts_hyperparams, seed=seed)

    # save results
    save_results_baseline_ml_hgb(opts_cv,  opts_score, opts_hyperparams, seed,
                                 y_true_repeat,
                                 y_pred_unclbr_repeat, proba_pred_unclbr_repeat, confus_mtx_unclbr_repeat, score_unclbr_repeat,
                                 y_pred_calibr_repeat, proba_pred_calibr_repeat, confus_mtx_calibr_repeat, score_calibr_repeat)

    print('Done!')
