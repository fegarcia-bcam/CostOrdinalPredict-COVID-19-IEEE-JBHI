import os
import json

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error, brier_score_loss
from imblearn.metrics import geometric_mean_score
from calc_scoring import get_confusion_mtx, calc_score

import config


# paths and files
PATH = config.PATH_RESULTS_CALIBRATE
FILENAME = 'results_baseline-clin.json'

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


def eval_performance(y_true, y_pred, proba_pred):
    # compute the list of performance metrics
    accurac = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    auc_aggre = roc_auc_score(y_true, proba_pred, multi_class='ovr', average='macro')
    auc_multi = roc_auc_score(y_true, proba_pred, multi_class='ovr', average=None)
    ap_aggre = average_precision_score(y_true, proba_pred, average='macro')
    ap_multi = average_precision_score(y_true, proba_pred, average=None)
    gms = geometric_mean_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    brier = 0
    for idx_l, label in enumerate(config.LABELS):
        brier_label = brier_score_loss((y_true == label), proba_pred[:, idx_l])
        brier += brier_label

    confus_mtx = get_confusion_mtx(y_true, y_pred, labels=config.LABELS)
    num_samples = confus_mtx.sum(axis=None)
    tot_cost_mtx = np.multiply(confus_mtx, config.COST_MTX)
    tot_cost_sum = tot_cost_mtx.sum(axis=None)
    cost_avg = tot_cost_sum / num_samples

    score = calc_score(y_true=y_true, y_pred=y_pred, proba_pred=proba_pred,
                       labels=config.LABELS, cost_mtx=config.COST_MTX,
                       scorer_name=OPTS_SCORE['scorer_name'], extra_args=OPTS_SCORE['scorer_args'])

    # gather results in a structured manner
    dict_perform = {'accurac': accurac,
                    'bal_acc': bal_acc,
                    'auc_aggre': auc_aggre,
                    'auc_multi': dict(zip(config.CLASS_NAMES, auc_multi.tolist())),
                    'ap_aggre': ap_aggre,
                    'ap_multi': dict(zip(config.CLASS_NAMES, ap_multi.tolist())),
                    'gms': gms,
                    'mae': mae,
                    'brier': brier,
                    'cost_avg': cost_avg,
                    'score': score}
    return dict_perform


if __name__ == '__main__':
    l_methods = ['PSI', 'Xie', 'NEWS2', 'PRIEST', 'ISARIC4C']
    for method in l_methods:
        # read from disk via json
        path = config.PATH_RESULTS_BASELINE
        filename = config.FILE_RESULTS_BASELINE.format('baseline-clin_{}'.format(method))

        f_results = os.path.join(path, filename)
        with open(f_results, 'r') as f_json_results:
            dict_results = json.load(f_json_results)

        # extract relevant info
        y_true = np.asarray(dict_results['y_true'])
        y_pred = np.asarray(dict_results['y_pred'])
        proba_pred = np.asarray(dict_results['proba_pred'])

        # evaluate performance
        dict_perform = eval_performance(y_true, y_pred, proba_pred)

    print('Done!')
