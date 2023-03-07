import os
import json

from datetime import datetime

import pandas as pd

from run_shap_expl import run_shap_explanation

import config


def load_data():
    df_data = pd.read_csv(config.FILE_DATA_IN, header=0, sep=config.DELIMITER)
    if len(df_data.columns) == 1:
        raise SystemExit(config.MSSG_ERROR_DATA)

    df_data = df_data.drop(columns=config.VARS_EXTRA)

    return df_data


def choose_scenario():
    # choose decision system
    decisor = 'map'
    # decisor = 'bayes'

    # choose nested cross validation scenario
    # opts_cv = {'n_iters': 10,
    #            'n_folds_out': 5,
    #            'n_folds_in': 5,
    #            'n_folds_fs': 2}  # quick calibration exploration
    opts_cv = {'n_iters': 20,
               'n_folds_out': 10,
               'n_folds_in': 10,
               'n_folds_fs': 5}  # exhaustive performance assessment

    # choose target score
    # opts_score = {'scorer_name': 'roc_auc',
    #               'scorer_args': {'average': 'macro',  # for multiclass: 'macro', 'weighted'
    #                               'multi_class': 'ovo'}}
    # opts_score = {'scorer_name': 'geo_mean_sens',
    #               'scorer_args': {'average': 'multiclass',
    #                               'correction': 10.0 ** (-1 * config.NUM_CLASSES)}}
    # opts_score = {'scorer_name': 'cost_score',
    #               'scorer_args': {'average': 'micro',  # 'micro', 'macro'
    #                               'reference': 'base'}}  # 'base', 'max'
    # opts_score = {'scorer_name': 'cost_score',
    #               'scorer_args': {'average': 'macro',  # 'micro', 'macro'
    #                               'reference': 'base'}}  # 'base', 'max'
    # opts_score = {'scorer_name': 'custom',
    #               'scorer_args': {'overall_average': 'harm',  # 'arithm', 'geom', 'harm'
    #                               'list_scorer_names': ['cost_score',
    #                                                     'cost_score'],
    #                               'list_extra_args': [{'average': 'micro', 'reference': 'base'},
    #                                                   {'average': 'macro', 'reference': 'base'}]
    #                               }
    #               }
    opts_score = {'scorer_name': 'custom',
                  'scorer_args': {'overall_average': 'harm',  # 'arithm', 'geom', 'harm'
                                  'list_scorer_names': ['cost_score',
                                                        'geo_mean_sens'],
                                  'list_extra_args': [{'average': 'micro', 'reference': 'base'},
                                                      {'average': 'multiclass', 'correction': 10.0 ** (-1 * config.NUM_CLASSES)}]
                                  }
                  }
    # opts_score = {'scorer_name': 'custom',
    #               'scorer_args': {'overall_average': 'harm',  # 'arithm', 'geom', 'harm'
    #                               'list_scorer_names': ['cost_score',
    #                                                     'geo_mean_sens'],
    #                               'list_extra_args': [{'average': 'macro', 'reference': 'base'},
    #                                                   {'average': 'multiclass', 'correction': 10.0 ** (-1 * config.NUM_CLASSES)}]
    #                               }
    #               }
    # opts_score = {'scorer_name': 'custom',
    #               'scorer_args': {'overall_average': 'harm',  # 'arithm', 'geom', 'harm'
    #                               'list_scorer_names': ['cost_score',
    #                                                     'cost_score',
    #                                                     'geo_mean_sens'],
    #                               'list_extra_args': [{'average': 'micro', 'reference': 'base'},
    #                                                   {'average': 'macro', 'reference': 'base'},
    #                                                   {'average': 'multiclass', 'correction': 10.0 ** (-1 * config.NUM_CLASSES)}]
    #                               }
    #               }

    # choose pipeline component stages
    opts_pipe = {'encoder': 'onehot',  # 'onehot', 'ordinal'
                 'scaler': 'robust',  # 'none', 'standard', 'robust'
                 'imputer': 'none',  # 'none', 'simple', 'knn', 'iterative'
                 'feat_sel': {'method': 'none',
                              'scoring': None
                              # 'scoring': {'scorer_name': 'geo_mean_sens',
                              #             'scorer_args': {'average': 'multiclass',
                              #                             'correction': 10.0 ** (-1 * config.NUM_CLASSES)}}
                              },
                 'decompos': {'type': 'ordered-part',  # 'regress', 'one-vs-one', 'ordered-part'
                              'policy': 'cost'  # 'equal', 'cost'
                              },
                 'balancer': {'ovs': 'random',  # 'none', 'random', 'smote', 'smote_border', 'adasyn'
                              'uns_clean': 'none',  # 'none', 'enn', 'tomek'
                              'uns_ctrl': 'random'  # 'none', 'random', 'cluster_c', 'near_miss'
                              },
                 'estimator': 'hgb'  # 'linear', 'svm', 'mlp', 'rf', 'ab', 'gb', 'hgb'
                 }

    # choose hyperparams (fixed!)
    hyperparams = {'encoder': {},  # encoder must never have hyperparams
                # ---------------------------------------------------------------- #
                'scaler': {},  # empty dict for defaults
                # ---------------------------------------------------------------- #
                'imputer': {},  # no imputer

                # 'imputer': {'add_indicator': False,
                #             'strategy': 'median'},  # SimpleImputer
                # 'imputer': {'add_indicator': True,
                #             'strategy': 'median'},  # SimpleImputer

                # 'imputer': {'add_indicator': False,
                #             'n_neighbors': 5,
                #             'weights': 'distance'},  # KNNImputer
                # 'imputer': {'add_indicator': True,
                #             'n_neighbors': 5,
                #             'weights': 'distance'},  # KNNImputer
                # 'imputer': {'add_indicator': False,
                #             'n_neighbors': 9,
                #             'weights': 'distance'},  # KNNImputer
                # 'imputer': {'add_indicator': True,
                #             'n_neighbors': 9,
                #             'weights': 'distance'},  # KNNImputer

                # 'imputer': {'add_indicator': False,
                #             'n_nearest_features': 4,
                #             'initial_strategy': 'median',
                #             'sample_posterior': True},  # IterativeImputer with BayesianRidge estimator
                # 'imputer': {'add_indicator': True,
                #             'n_nearest_features': 4,
                #             'initial_strategy': 'median',
                #             'sample_posterior': True},  # IterativeImputer with BayesianRidge estimator
                # 'imputer': {'add_indicator': False,
                #             'n_nearest_features': 8,
                #             'initial_strategy': 'median',
                #             'sample_posterior': True},  # IterativeImputer with BayesianRidge estimator
                # 'imputer': {'add_indicator': True,
                #             'n_nearest_features': 8,
                #             'initial_strategy': 'median',
                #             'sample_posterior': True},  # IterativeImputer with BayesianRidge estimator
                # ---------------------------------------------------------------- #
                'feat_sel': {},  # no feature selection

                # 'feat_sel': {'num_feats': 20, 'num_neighbs': 3},  # 'filt-mutual_info_regress'

                # 'feat_sel': {'num_feats': 20, 'num_neighbs': 3},  # 'filt-mutual_info_classif'

                # 'feat_sel': {'num_feats': 20, 'method': 'MIQ'},  # 'filt-mrmr'
                # 'feat_sel': {'num_feats': 20, 'method': 'MID'},  # 'filt-mrmr'

                # 'feat_sel': {'num_feats': 20, 'num_neighbs': 10, 'discr_thresh': 2, 'n_jobs': config.NUM_JOBS_RELIEF},  # 'filt-relieff'
                # 'feat_sel': {'num_feats': 20, 'num_neighbs': 100, 'discr_thresh': 2, 'n_jobs': config.NUM_JOBS_RELIEF},  # 'filt-relieff'

                # 'feat_sel': {'num_feats': 20, 'discr_thresh': 2, 'n_jobs': config.NUM_JOBS_RELIEF},  # 'filt-multisurf'

                # 'feat_sel': {'alpha': 5.0e-2},  # 'embed-l1_regress'

                # 'feat_sel': {'C': 2.0e-2, 'n_jobs': config.NUM_JOBS_LOGIST_REG},  # 'embed-l1_classif'

                # 'feat_sel': {'alpha': 1.0e-2, 'num_feats': 20, 'step': 2},  # 'wrap-rfe_regress'

                # 'feat_sel': {'C': 1.0e-3, 'num_feats': 20, 'step': 2},  # 'wrap-rfe_classif'

                # 'feat_sel': {'alpha': 1.0e-2,
                #              'min_feats': 5, 'step': 5,
                #              'num_cv_folds': opts_cv['n_folds_fs']},  # 'wrap-rfecv_regress'

                # 'feat_sel': {'C': 1.0e-3,
                #              'min_feats': 5, 'step': 5,
                #              'num_cv_folds': opts_cv['n_folds_fs']},  # 'wrap-rfecv_classif'

                # 'feat_sel': {'method': 'hgb',
                #              'num_feats': 20, 'direction': 'forward',
                #              'num_cv_folds': opts_cv['n_folds_fs']},  # 'wrap-seq_regress' and 'wrap-seq_classif'

                # 'feat_sel': {'method': 'hgb',
                #              'support_categ_feats': False,
                #              'penaliz': 0.2,
                #              'num_iters': 1000,
                #              'num_indivs': 100,
                #              'options': {'prob_flip': 0.10, 'prob_cross': 0.70, 'prob_mutat': 0.020,
                #                          'select_type': 'roulette', 'select_param': None},
                #              'num_cv_folds': opts_cv['n_folds_fs']},  # 'wrap-ga_regress' and 'wrap-ga_classif'
                # 'feat_sel': {'method': 'hgb',
                #              'support_categ_feats': False,
                #              'penaliz': 0.2,
                #              'num_iters': 1000,
                #              'num_indivs': 100,
                #              'options': {'prob_flip': 0.10, 'prob_cross': 0.70, 'prob_mutat': 0.001,
                #                          'select_type': 'roulette', 'select_param': None},
                #              'num_cv_folds': opts_cv['n_folds_fs']},  # 'wrap-ga_regress' and 'wrap-ga_classif'

                # 'feat_sel': {'method': 'hgb',
                #              'support_categ_feats': False,
                #              'penaliz': 0.2,
                #              'num_iters': 2000,
                #              'num_parts': 30,
                #              'options': {'c1': 0.5, 'c2': 0.5, 'w': 0.6, 'k': 30, 'p': 1},
                #              'vel_clamp': (-2.0, 2.0),
                #              'num_cv_folds': opts_cv['n_folds_fs']},  # 'wrap-pso_regress' and 'wrap-pso_classif'
                # 'feat_sel': {'method': 'hgb',
                #              'support_categ_feats': False,
                #              'penaliz': 0.2,
                #              'num_iters': 2000,
                #              'num_parts': 30,
                #              'options': {'c1': 0.5, 'c2': 0.5, 'w': 0.6, 'k': 30, 'p': 1},
                #              'vel_clamp': (-6.0, 6.0),
                #              'num_cv_folds': opts_cv['n_folds_fs']},  # 'wrap-pso_regress' and 'wrap-pso_classif'
                # ---------------------------------------------------------------- #
                'decompos': {},  # decomposer must never have hyperparams
                # ---------------------------------------------------------------- #
                # RandomOverSampler
                'balancer': {'ovs': {},  # empty dicts for defaults
                             'uns_clean': {},  # no cleaning
                             'uns_ctrl': {}  # RandomUnderSampler, empty dicts for defaults
                             },

                # SMOTE
                # 'balancer': {'ovs': {},  # empty dicts for defaults
                #              'uns_clean': {},  # Tomek cleaning, empty dicts for defaults
                #              'uns_ctrl': {}  # RandomUnderSampler, empty dicts for defaults
                #              },

                # Borderline-SMOTE, kind-1
                # 'balancer': {'ovs': {'kind': 'borderline-1'},
                #              'uns_clean': {},  # no cleaning
                #              'uns_ctrl': {}  # RandomUnderSampler, empty dicts for defaults
                #              },

                # ADASYN
                # 'balancer': {'ovs': {},  # empty dicts for defaults
                #              'uns_clean': {},  # no cleaning
                #              'uns_ctrl': {}  # RandomUnderSampler, empty dicts for defaults
                #              },
                # ---------------------------------------------------------------- #
                # Linear models: Ridge (L2) | LogisticRegression (L2)
                # 'estimator': {'alpha': 2.0,
                #               'solver': 'saga',
                #               'max_iter': int(1e4)}  # linear-regress
                # 'estimator': {'C': 20.0,
                #               'penalty': 'l2',
                #               'multi_class': 'auto',
                #               'solver': 'saga',
                #               'max_iter': int(1e4),
                #               'n_jobs': config.NUM_JOBS_LOGIST_REG}  # linear-classif

                # Support Vector Machines
                # 'estimator': {'kernel': 'rbf',
                #               'gamma': 1.0e-4,
                #               'C': 1.0e+1}  # SVM-regress
                # 'estimator': {'kernel': 'rbf',
                #               'gamma': 1.0e-4,
                #               'C': 1.0e+1,
                #               'probability': True}  # SVM-classif

                # Multi-Layer Perceptron
                # 'estimator': {'hidden_layer_sizes': [64, 32, 4],
                #               'alpha': 1e+2,
                #               'solver': 'lbfgs',
                #               'max_iter': int(1e4)}  # MLP-regress and MLP-classif

                # Random Forest
                # 'estimator': {'n_estimators': 100,
                #               'criterion': 'mae',  # 'mae' will be deprecated; substitute by 'absolute_error'
                #               'max_features': 0.20,
                #               'max_depth': None}  # RandomForest-regress
                # 'estimator': {'n_estimators': 100,
                #               'criterion': 'mse',  # 'mse' will be deprecated; substitute by 'squared_error'
                #               'max_features': 0.20,
                #               'max_depth': None}  # RandomForest-regress
                # 'estimator': {'n_estimators': 100,
                #               'criterion': 'gini',
                #               'max_features': 0.20,
                #               'max_depth': None}  # RandomForest-classif
                # 'estimator': {'n_estimators': 100,
                #               'criterion': 'entropy',
                #               'max_features': 0.20,
                #               'max_depth': None}  # RandomForest-classif

                # AdaBoost currently not a priority
                # 'estimator': {'n_estimators': 100,
                #               'base_estim': 'decision_tree',
                #               'max_depth': 3}  # AdaBoost-regress and AdaBoost-classif

                # GradBoost currently not a priority
                # 'estimator': {'loss': 'ls',
                #               'learning_rate': 0.10,
                #               'n_estimators': 100,
                #               'max_features': 0.20}  # GradBoost-regress
                # 'estimator': {'loss': 'deviance',
                #               'learning_rate': 0.10,
                #               'n_estimators': 100,
                #               'max_features': 0.20}  # GradBoost-classif

                # Histogram Gradient Boosting
                # 'estimator': {'loss': 'least_squares',
                #               'learning_rate': 0.10,
                #               'max_iter': 100,
                #               'max_leaf_nodes': None,
                #               'max_depth': None,
                #               'support_categ_feats': False}  # HistGradBoost-regress
                # 'estimator': {'loss': 'log_loss',
                #               'learning_rate': 0.10,
                #               'max_iter': 100,
                #               'max_leaf_nodes': None,
                #               'max_depth': None,
                #               'support_categ_feats': True}  # HistGradBoost-classif (with categorical feats)
                # 'estimator': {'loss': 'log_loss',
                #               'learning_rate': 0.10,
                #               'max_iter': 100,
                #               'max_leaf_nodes': None,
                #               'max_depth': None,
                #               'support_categ_feats': False}  # HistGradBoost-classif
                # 'estimator': {'loss': 'least_squares',
                #               'learning_rate': 0.10,
                #               'max_iter': 100,
                #               'max_leaf_nodes': None,
                #               'max_depth': None,
                #               'support_categ_feats': True}  # HistGradBoost-regress (with categorical feats)

                'estimator': {'loss': 'log_loss',
                              'learning_rate': 0.20,
                              'max_iter': 200,
                              'max_leaf_nodes': None,
                              'max_depth': None,
                              'support_categ_feats': False}  # HistGradBoost-classif. extended param options
                }

    return decisor, opts_cv, opts_score, opts_pipe, hyperparams


def save_results_shap(decisor, opts_cv, opts_score, opts_pipe, hyperparams, seed, shap_values):
    # create dict containing all relevant info
    results = {'decisor': decisor, 'opts_cv': opts_cv, 'opts_score': opts_score,
               'opts_pipe': opts_pipe, 'hyperparams': hyperparams, 'seed': seed,
               'shap_values': shap_values}

    # generate timestamp and a file in the desired path
    path = config.PATH_RESULTS_SHAP
    if not os.path.exists(path):
        os.makedirs(path)
    timestamp = datetime.now().strftime(config.DATETIME_FORMAT)
    filename = config.FILE_RESULTS_SHAP.format(timestamp)
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
    decisor, opts_cv, opts_score, opts_pipe, hyperparams = choose_scenario()

    # run SHAPLY additive explanation of feature contributions
    seed = config.SEED_CLASSIFY
    shap_values = run_shap_explanation(df_data, decisor, opts_cv, opts_score, opts_pipe, hyperparams, seed=seed)

    # save results
    save_results_shap(decisor, opts_cv, opts_score, opts_pipe, hyperparams, seed, shap_values)

    print('Done!')
