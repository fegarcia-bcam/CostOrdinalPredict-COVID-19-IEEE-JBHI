import numpy as np
from scipy.stats import gmean, hmean

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, balanced_accuracy_score, top_k_accuracy_score, f1_score, fbeta_score
from sklearn.metrics import roc_auc_score, matthews_corrcoef, cohen_kappa_score

from imblearn.metrics import sensitivity_score, specificity_score, geometric_mean_score

from utils_format import check_labels, check_classes
from utils_cost import check_confusion_matrix, calc_cost_score

SCORERS_AVAIL = ['accuracy', 'balanc_accur', 'top_k_accur', 'f1', 'fbeta', 'roc_auc', 'matthews', 'cohen',
                 'sensit', 'specif', 'geo_mean_sens', 'cost_score', 'custom']
SCORE_AVERAGES = ['arithm', 'geom', 'harm']


# reciprocal to obtaining the confusion matrix from classes
def get_classes_from_confusion_mtx(confus_mtx, labels):
    labels = check_labels(labels).to_numpy()
    num_classes = labels.size
    confus_mtx = check_confusion_matrix(confus_mtx, num_classes=num_classes)

    idx_true = []
    idx_pred = []
    for i in range(num_classes):
        for j in range(num_classes):
            num_cases = confus_mtx[i, j]
            idx_true += [i] * num_cases
            idx_pred += [j] * num_cases

    y_true = labels[idx_true]
    y_pred = labels[idx_pred]
    y_true = check_classes(y_true, num_classes=num_classes)
    y_pred = check_classes(y_pred, num_classes=num_classes)

    return y_true, y_pred


def get_confusion_mtx(y_true, y_pred, labels):
    confus_mtx = confusion_matrix(y_true, y_pred, labels=labels)
    return confus_mtx


def calc_score(scorer_name, y_true, y_pred, proba_pred, labels, cost_mtx, extra_args):
    if isinstance(scorer_name, str) and isinstance(extra_args, dict):
        score = calc_score_single(scorer_name, y_true, y_pred, proba_pred, labels, cost_mtx, extra_args)
    elif isinstance(scorer_name, list) and isinstance(extra_args, list):
        score = calc_score_multi(scorer_name, y_true, y_pred, proba_pred, labels, cost_mtx, extra_args)
    else:
        raise TypeError
    return score


def calc_score_single(scorer_name, y_true, y_pred, proba_pred, labels, cost_mtx, extra_args):
    score = None
    if not isinstance(extra_args, dict):
        raise TypeError

    if scorer_name not in SCORERS_AVAIL:
        raise NotImplementedError

    elif scorer_name == 'accuracy':
        score = accuracy_score(y_true, y_pred, **extra_args)

    elif scorer_name == 'balanc_accur':
        score = balanced_accuracy_score(y_true, y_pred, **extra_args)

    elif scorer_name == 'top_k_accur':
        score = top_k_accuracy_score(y_true, proba_pred, labels=labels, **extra_args)

    elif scorer_name == 'f1':
        score = f1_score(y_true, y_pred, labels=labels, **extra_args)

    elif scorer_name == 'fbeta':
        if 'beta' not in extra_args:
            raise ValueError
        beta = extra_args.pop('beta')
        score = fbeta_score(y_true, y_pred, beta=beta, labels=labels, **extra_args)

    elif scorer_name == 'roc_auc':
        score = roc_auc_score(y_true, proba_pred, labels=labels, **extra_args)

    elif scorer_name == 'matthews':
        score = matthews_corrcoef(y_true, y_pred)

    elif scorer_name == 'cohen':
        score = cohen_kappa_score(y_true, y_pred, labels=labels, **extra_args)

    elif scorer_name == 'sensit':
        if 'average' not in extra_args:
            raise ValueError
        average = extra_args['average']
        if average == 'min':  # custom, taken from ORCA project
            sensit_all = sensitivity_score(y_true, y_pred, labels=labels, average=None)
            score = sensit_all.min()
        else:
            score = sensitivity_score(y_true, y_pred, labels=labels, average=average)

    elif scorer_name == 'specif':
        if 'average' not in extra_args:
            raise ValueError
        average = extra_args['average']
        if average == 'min':  # custom, taken from ORCA project
            specif_all = specificity_score(y_true, y_pred, labels=labels, average=None)
            score = specif_all.min()
        else:
            score = specificity_score(y_true, y_pred, labels=labels, average=average)

    elif scorer_name == 'geo_mean_sens':
        score = geometric_mean_score(y_true, y_pred, labels=labels, **extra_args)

    elif scorer_name == 'cost_score':
        confus_mtx = get_confusion_mtx(y_true, y_pred, labels)
        score = calc_cost_score(confus_mtx, cost_mtx, **extra_args)

    elif scorer_name == 'custom':
        if 'overall_average' not in extra_args:
            raise ValueError
        if 'list_scorer_names' not in extra_args:
            raise ValueError
        if 'list_extra_args' not in extra_args:
            raise ValueError

        overall_average = extra_args['overall_average']
        list_scorer_names = extra_args['list_scorer_names']
        list_extra_args = extra_args['list_extra_args']

        # compute each of the custom scores
        score_custom = calc_score(list_scorer_names, y_true, y_pred, proba_pred, labels, cost_mtx, list_extra_args)
        score_custom = np.asarray(list(score_custom.values()))

        score = 0.0
        # average them
        if overall_average not in SCORE_AVERAGES:
            raise ValueError
        elif overall_average == 'arithm':
            score = score_custom.mean()
        elif overall_average == 'geom':
            if ~np.isclose(score_custom, 0.0).any():  # avoids RuntimeWarning in geometric mean
                score = gmean(score_custom)
        elif overall_average == 'harm':
            if ~np.isclose(score_custom, 0.0).any():  # avoids RuntimeWarning in harmonic mean
                score = hmean(score_custom)

    return score


def calc_score_multi(scorer_name, y_true, y_pred, proba_pred, labels, cost_mtx, extra_args):
    score = {}
    for sc_name, sc_extra_args in zip(scorer_name, extra_args):
        sc = calc_score_single(sc_name, y_true, y_pred, proba_pred, labels, cost_mtx, sc_extra_args)
        score[sc_name] = sc

    return score
