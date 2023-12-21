import os
import json

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error, brier_score_loss
from imblearn.metrics import geometric_mean_score

import config


# statistics
ALPHA = 0.05

# paths and files
# PATH = config.PATH_RESULTS_CALIBRATE
# FILENAME = 'results_final.json'
# TAGS = ['_unclbr', '_calibr']
# PATH = config.PATH_RESULTS_BASELINE
# FILENAME = 'results_baseline-ml_HGB.json'
# TAGS = ['_unclbr', '_calibr']
PATH = config.PATH_RESULTS_BASELINE
FILENAME = 'results_baseline-ml_Carr.json'
TAGS = ['']


def eval_performance(l_y_true, l_y_pred, l_proba_pred, l_confus_mtx, l_score):
    # compute the list of performance metrics
    l_accurac, l_bal_acc = [], []
    l_auc_aggre, l_auc_multi, l_ap_aggre, l_ap_multi = [], [], [], []
    l_gms, l_mae, l_brier, l_cost_avg = [], [], [], []
    for y_true, y_pred, proba_pred, confus_mtx in zip(l_y_true, l_y_pred, l_proba_pred, l_confus_mtx):
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

        num_samples = confus_mtx.sum(axis=None)
        tot_cost_mtx = np.multiply(confus_mtx, config.COST_MTX)
        tot_cost_sum = tot_cost_mtx.sum(axis=None)
        cost_avg = tot_cost_sum / num_samples

        l_accurac.append(accurac)
        l_bal_acc.append(bal_acc)
        l_auc_aggre.append(auc_aggre)
        l_auc_multi.append(auc_multi)
        l_ap_aggre.append(ap_aggre)
        l_ap_multi.append(ap_multi)
        l_gms.append(gms)
        l_mae.append(mae)
        l_brier.append(brier)
        l_cost_avg.append(cost_avg)

    l_accurac = pd.Series(l_accurac)
    l_bal_acc = pd.Series(l_bal_acc)
    l_auc_aggre = pd.Series(l_auc_aggre)
    l_auc_multi = pd.DataFrame(data=l_auc_multi, columns=config.CLASS_NAMES)
    l_ap_aggre = pd.Series(l_ap_aggre)
    l_ap_multi = pd.DataFrame(data=l_ap_multi, columns=config.CLASS_NAMES)
    l_gms = pd.Series(l_gms)
    l_mae = pd.Series(l_mae)
    l_score = pd.Series(l_score)
    l_brier = pd.Series(l_brier)
    l_cost_avg = pd.Series(l_cost_avg)

    # compute the quantiles
    q_md = 0.5
    q_lo = ALPHA / 2.0
    q_up = 1.0 - q_lo

    q_accurac = l_accurac.quantile(q=[q_md, q_lo, q_up])
    q_bal_acc = l_bal_acc.quantile(q=[q_md, q_lo, q_up])
    q_auc_aggre = l_auc_aggre.quantile(q=[q_md, q_lo, q_up])
    q_auc_multi = l_auc_multi.quantile(q=[q_md, q_lo, q_up])
    q_ap_aggre = l_ap_aggre.quantile(q=[q_md, q_lo, q_up])
    q_ap_multi = l_ap_multi.quantile(q=[q_md, q_lo, q_up])
    q_gms = l_gms.quantile(q=[q_md, q_lo, q_up])
    q_mae = l_mae.quantile(q=[q_md, q_lo, q_up])
    q_score = l_score.quantile(q=[q_md, q_lo, q_up])
    q_brier = l_brier.quantile(q=[q_md, q_lo, q_up])
    q_cost_avg = l_cost_avg.quantile(q=[q_md, q_lo, q_up])

    # gather results in a structured manner
    dict_perform = {'accurac': {'all': l_accurac.tolist(), 'quants': q_accurac.to_dict()},
                    'bal_acc': {'all': l_bal_acc.tolist(), 'quants': q_bal_acc.to_dict()},
                    'auc_aggre': {'all': l_auc_aggre.tolist(), 'quants': q_auc_aggre.to_dict()},
                    'auc_multi': {'all': l_auc_multi.to_dict(orient='list'), 'quants': q_auc_multi.to_dict()},
                    'ap_aggre': {'all': l_ap_aggre.tolist(), 'quants': q_ap_aggre.to_dict()},
                    'ap_multi': {'all': l_ap_multi.to_dict(orient='list'), 'quants': q_ap_multi.to_dict()},
                    'gms': {'all': l_gms.tolist(), 'quants': q_gms.to_dict()},
                    'mae': {'all': l_mae.tolist(), 'quants': q_mae.to_dict()},
                    'brier': {'all': l_brier.tolist(), 'quants': q_brier.to_dict()},
                    'cost_avg': {'all': l_cost_avg.tolist(), 'quants': q_cost_avg.to_dict()},
                    'score': {'all': l_score.tolist(), 'quants': q_score.to_dict()}}
    return dict_perform


if __name__ == '__main__':
    # read from disk via json
    f_results = os.path.join(PATH, FILENAME)
    with open(f_results, 'r') as f_json_results:
        dict_results = json.load(f_json_results)

    for tag in TAGS:
        # extract relevant info
        num_repeats = dict_results['opts_cv']['n_iters']
        l_y_true = dict_results['y_true_repeat']
        l_y_pred = dict_results['y_pred{}_repeat'.format(tag)]
        l_proba_pred = dict_results['proba_pred{}_repeat'.format(tag)]
        l_confus_mtx = dict_results['confus_mtx{}_repeat'.format(tag)]
        l_score = dict_results['score{}_repeat'.format(tag)]

        # reformat
        if len(l_y_true) != num_repeats:
            raise RuntimeError
        l_y_true = [np.asarray(y_true) for y_true in l_y_true]

        if len(l_y_pred) != num_repeats:
            raise RuntimeError
        l_y_pred = [np.asarray(y_pred) for y_pred in l_y_pred]

        if len(l_proba_pred) != num_repeats:
            raise RuntimeError
        l_proba_pred = [np.asarray(proba_pred) for proba_pred in l_proba_pred]

        if len(l_confus_mtx) != num_repeats:
            raise RuntimeError
        l_confus_mtx = [np.asarray(confus_mtx) for confus_mtx in l_confus_mtx]

        if len(l_score) != num_repeats:
            raise RuntimeError
        l_score = np.asarray(l_score)

        # evaluate performance
        d_perf = eval_performance(l_y_true, l_y_pred, l_proba_pred, l_confus_mtx, l_score)

    print('Done!')
