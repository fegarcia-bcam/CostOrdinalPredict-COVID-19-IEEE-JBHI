import os
import json

from datetime import datetime

import numpy as np
import pandas as pd

from scipy.stats.mstats import winsorize

from numpy.random import SeedSequence

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, GridSearchCV

from baseline_clin import NEWS2_COLS_IN, NEWS2_CATEGOR
from baseline_clin import get_clinical_score_news2, load_data, impute_data

from calc_scoring import get_confusion_mtx, calc_score

import config


# adaptation from
# Carr et al. 2021, BMC Med || doi:10.1186/s12916-020-01893-3
# input information
CARR_COLS_IN = NEWS2_COLS_IN + \
               ['pat_age',
                # the indicator flag for supplemental oxygen flow substitutes its exact rate
                'blood-t_crp[log10]', 'blood-t_urea[log10]',
                'blood-t_creatin[log10]',
                # creatinine substitutes eGFR as a biomarker of renal function
                # Zhang et al. 2019, Annu Rev Physiol || doi:10.1146/annurev-physiol-020518-114605
                'blood-t_neutroph[log10]', 'blood-t_neutro-lympho[log10]']
CARR_COLS_AUX = ['pat_age', 'emerg-status_news2', 'emerg-status_spo2', 'emerg-status_extra-o2',
                 'blood-t_crp[log10]', 'blood-t_urea[log10]', 'blood-t_creatin[log10]',
                 'blood-t_neutroph[log10]', 'blood-t_neutro-lympho[log10]']
CARR_COLS_OUT = ['pat_age', 'emerg-status_news2', 'emerg-status_spo2', 'emerg-status_extra-o2',
                 'blood-t_crp[sqrt]', 'blood-t_urea[sqrt]', 'blood-t_creatin[sqrt]',
                 'blood-t_neutroph[sqrt]', 'blood-t_neutro-lympho[ln]']
# we exclude nosocomial indicator flag, because here all patients were non-nosocomial by the inclusion criteria

CARR_CATEGOR = {'emerg-status_extra-o2': {0: 0, 1: 1}}  # 0: False, 1: True


# general setup
# choose nested cross validation scenario
OPTS_CV = {'n_iters': 20,
           'n_folds_out': 10,
           'n_folds_in': 10}

# custom performance score
OPTS_SCORE = {'scorer_name': 'custom',
              'scorer_args': {'overall_average': 'harm',  # 'arithm', 'geom', 'harm'
                              'list_scorer_names': ['cost_score',
                                                    'geo_mean_sens'],
                              'list_extra_args': [{'average': 'micro', 'reference': 'base'},
                                                  {'average': 'multiclass',
                                                   'correction': 10.0 ** (-1 * config.NUM_CLASSES)}]
                              }
              }

SEED = config.SEED_CLASSIFY

# VERBOSE = 1
VERBOSE = 4  # maximum verbosity

# NUM_JOBS = None
NUM_JOBS = -1


# pre-processing: scaling
WINSOR_QUANT = 0.01

SCALER = StandardScaler()

# imputation
IMPUTER_NEIGHBORS = 9
IMPUTER_WEIGHTS = 'distance'
# IMPUTER_NEIGHBORS = 5
# IMPUTER_WEIGHTS = 'uniform'

IMPUTER = KNNImputer(n_neighbors=IMPUTER_NEIGHBORS, weights=IMPUTER_WEIGHTS)

# classification
LOGIST_REGR_PENALTY = 'l2'
LOGIST_REGR_SOLVER = 'lbfgs'
LOGIST_REGR_MULTICLASS = 'multinomial'

CLASSIF_BASE = LogisticRegression(penalty=LOGIST_REGR_PENALTY,
                                  solver=LOGIST_REGR_SOLVER, multi_class=LOGIST_REGR_MULTICLASS)

CLASSIF_PIPE = Pipeline([('scaler', clone(SCALER)),
                         ('imputer', clone(IMPUTER)),
                         ('classifier', clone(CLASSIF_BASE))])

# hyperparameter tuning
HYPERPARAM_GRID_NUM = 25
HYPERPARAM_GRID_LOG_MIN = -4.0
HYPERPARAM_GRID_LOG_MAX = +4.0
HYPERPARAM_GRID = {'classifier__C': np.logspace(HYPERPARAM_GRID_LOG_MIN, HYPERPARAM_GRID_LOG_MAX, num=HYPERPARAM_GRID_NUM, endpoint=True).tolist()}
HYPERPARAM_SCORE = 'roc_auc_ovr'
HYPERPARAM_REFIT = False

OPTS_HYPERPARAMS = {'search': 'grid', 'set': HYPERPARAM_GRID}

CLASSIF_TUNE = GridSearchCV(estimator=clone(CLASSIF_PIPE),
                            param_grid=HYPERPARAM_GRID, cv=OPTS_CV['n_folds_in'],
                            scoring=HYPERPARAM_SCORE, refit=HYPERPARAM_REFIT,
                            verbose=VERBOSE, n_jobs=NUM_JOBS)

# calibration
CALIBR_METHOD = 'sigmoid'
CALIBR_ENSEMBLE = True

CLASSIF_CALIBR = CalibratedClassifierCV(estimator=None,
                                        cv=OPTS_CV['n_folds_in'], method=CALIBR_METHOD, ensemble=CALIBR_ENSEMBLE)


def fit_predict_ml_carr(X, y):
    # train and evaluate
    y_true_repeat, y_pred_repeat, proba_pred_repeat = [], [], []
    confus_mtx_repeat, score_repeat = [], []

    # generate random sequence
    seed_seq = SeedSequence(SEED)
    l_random_states = seed_seq.generate_state(n_words=OPTS_CV['n_iters'])

    for idx_iter, random_state in enumerate(l_random_states):
        l_y_true, l_y_pred, l_proba_pred = [], [], []

        skf_cv = StratifiedKFold(n_splits=OPTS_CV['n_folds_out'], shuffle=True, random_state=random_state)
        for idx_cv, (idx_train, idx_test) in enumerate(skf_cv.split(X, y)):
            X_train = X.iloc[idx_train, :]
            y_train = y.iloc[idx_train]
            X_test = X.iloc[idx_test, :]
            y_test = y.iloc[idx_test]

            # tune hyperparameters
            classif_tune = clone(CLASSIF_TUNE)
            classif_tune.fit(X_train, y_train)
            if VERBOSE:
                print('Model trained (GridSearchCV): Fold {}/{} [Iteration {}/{}]'.format(idx_cv + 1, OPTS_CV['n_folds_out'], idx_iter + 1, OPTS_CV['n_iters']))

            # get the best hyperparameter choice
            classif_pipe = clone(CLASSIF_PIPE)
            classif_pipe.set_params(**classif_tune.best_params_)

            # calibrate
            classif_calibr = clone(CLASSIF_CALIBR)
            classif_calibr.set_params(**{'estimator': classif_pipe})
            classif_calibr.fit(X_train, y_train)
            if VERBOSE:
                print('Model trained (CalibrClassifCV): Fold {}/{} [Iteration {}/{}]'.format(idx_cv + 1, OPTS_CV['n_folds_out'], idx_iter + 1, OPTS_CV['n_iters']))

            # predict
            y_true_test = y_test.to_numpy()
            y_pred_test = classif_calibr.predict(X_test)
            proba_pred_test = classif_calibr.predict_proba(X_test)

            # concatenate intermediate results
            l_y_true += y_true_test.tolist()
            l_y_pred += y_pred_test.tolist()
            l_proba_pred += proba_pred_test.tolist()

        # compute confusion matrix and custom score
        confus_mtx = get_confusion_mtx(l_y_true, l_y_pred, labels=config.LABELS)
        score = calc_score(y_true=l_y_true, y_pred=l_y_pred, proba_pred=l_proba_pred,
                           labels=config.LABELS, cost_mtx=config.COST_MTX,
                           scorer_name=OPTS_SCORE['scorer_name'], extra_args=OPTS_SCORE['scorer_args'])

        # append results from this iteration
        y_true_repeat.append(l_y_true)
        y_pred_repeat.append(l_y_pred)
        proba_pred_repeat.append(l_proba_pred)
        confus_mtx_repeat.append(confus_mtx)
        score_repeat.append(score)

        if VERBOSE:
            print('Iteration {}/{} finished'.format(idx_iter + 1, OPTS_CV['n_iters']))

    return y_true_repeat, y_pred_repeat, proba_pred_repeat, confus_mtx_repeat, score_repeat


def save_results_baseline_ml_carr(y_true_repeat, y_pred_repeat, proba_pred_repeat,
                                  confus_mtx_repeat, score_repeat):
    # create dict containing all relevant info
    results = {'opts_cv': OPTS_CV, 'opts_score': OPTS_SCORE, 'OPTS_HYPERPARAMS': OPTS_HYPERPARAMS, 'seed': SEED,
               'y_true_repeat': y_true_repeat,
               'y_pred_repeat': y_pred_repeat,
               'proba_pred_repeat': proba_pred_repeat,
               'confus_mtx_repeat': [cm.tolist() for cm in confus_mtx_repeat],
               'score_repeat': score_repeat}

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
    # import data file(s)
    data_main = load_data(filename=config.FILE_DATA_IN)
    df_X = data_main.drop(columns=[config.VAR_CLASSIF])
    df_y = data_main[config.VAR_CLASSIF]

    data_suppl = load_data(filename=config.FILE_DATA_SUPPL)
    col_suppl = 'emerg-status_extra-o2'
    idx_loc = df_X.columns.get_loc('emerg-status_spo2') + 1
    df_X.insert(idx_loc, col_suppl, data_suppl[col_suppl])

    # retain only the strictly necessary information
    df_X_sel = df_X.copy()
    df_X_sel = df_X_sel[CARR_COLS_IN]

    # apply required transforms on inputs, including auxiliary variables
    # compute NEWS2 score
    df_X_aux = df_X_sel[NEWS2_COLS_IN]
    df_X_aux = impute_data(df_X_aux, replace_categ=NEWS2_CATEGOR)

    df_X_sel['emerg-status_news2'] = df_X_aux.apply(get_clinical_score_news2, axis='columns')
    df_X_sel = df_X_sel[CARR_COLS_AUX]

    # apply required transforms on inputs, including management of units and natural/logarithmic scales
    df_X_sel['blood-t_crp'] = np.power(10.0, df_X_sel['blood-t_crp[log10]'])  # undo log_10 logarithmic transform
    df_X_sel['blood-t_crp[sqrt]'] = np.sqrt(df_X_sel['blood-t_crp'])  # apply sqrt transform
    df_X_sel['blood-t_urea'] = np.power(10.0, df_X_sel['blood-t_urea[log10]'])  # undo log_10 logarithmic transform
    df_X_sel['blood-t_urea'] = (1 / 6.006) * df_X_sel['blood-t_urea']  # change units: from [mg/dL] to [mmol/L]
    df_X_sel['blood-t_urea[sqrt]'] = np.sqrt(df_X_sel['blood-t_urea'])  # apply sqrt transform
    df_X_sel['blood-t_creatin'] = np.power(10.0, df_X_sel['blood-t_creatin[log10]'])  # undo log_10 logarithmic transform
    df_X_sel['blood-t_creatin[sqrt]'] = np.sqrt(df_X_sel['blood-t_creatin'])  # apply sqrt transform
    df_X_sel['blood-t_neutroph'] = np.power(10.0, df_X_sel['blood-t_neutroph[log10]'])  # undo log_10 logarithmic transform
    df_X_sel['blood-t_neutroph'] = 1.0e-3 * df_X_sel['blood-t_neutroph']  # change units: from [10^6 U/L] to [10^9 U/L]
    df_X_sel['blood-t_neutroph[sqrt]'] = np.sqrt(df_X_sel['blood-t_neutroph'])  # apply sqrt transform
    df_X_sel['blood-t_neutro-lympho'] = np.power(10.0, df_X_sel['blood-t_neutro-lympho[log10]'])  # undo log_10 logarithmic transform
    df_X_sel['blood-t_neutro-lympho[ln]'] = np.sqrt(df_X_sel['blood-t_neutro-lympho'])  # change logarithm base to log_e

    # retain only the strictly necessary information
    df_X_sel = df_X_sel[CARR_COLS_OUT]

    # winsorize as in Carr et al.'s repository: https://github.com/ewancarr/NEWS2-COVID-19
    # even tough strictly (in theory), this transformation should be based only on the training set distribution!
    for col in df_X_sel.columns:
        if col not in ['pat_age', 'emerg-status_news2', 'emerg-status_extra-o2']:
            df_X_sel[col] = winsorize(df_X_sel[col], limits=[WINSOR_QUANT] * 2, nan_policy='propagate')

    # run repeated nested cross validation
    (y_true_repeat, y_pred_repeat, proba_pred_repeat,
     confus_mtx_repeat, score_repeat) = fit_predict_ml_carr(df_X_sel, df_y)

    # save results
    save_results_baseline_ml_carr(y_true_repeat, y_pred_repeat, proba_pred_repeat,
                                  confus_mtx_repeat, score_repeat)

    print('Done!')
