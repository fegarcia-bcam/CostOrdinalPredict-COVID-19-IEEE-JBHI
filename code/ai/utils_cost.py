import numpy as np

from utils_format import check_num_classes

COST_AVERAGE = ['micro', 'macro']
COST_AVG_DEFAULT = 'micro'
COST_REFERENCE = ['base', 'max']
COST_REF_DEFAULT = 'base'


def check_confusion_matrix(confus_mtx, num_classes=None):
    # check types
    if not isinstance(confus_mtx, np.ndarray):
        raise TypeError
    if num_classes is None:
        num_classes = confus_mtx.shape[0]
    else:
        num_classes = check_num_classes(num_classes)

    # check confusion requirements
    if confus_mtx.shape != (num_classes, num_classes):
        raise ValueError
    if np.any(confus_mtx < 0):
        raise ValueError
    confus_mtx = confus_mtx.astype(np.uint16, casting='unsafe')  # quick check for all non-negative integers

    return confus_mtx


def check_cost_matrix(cost_mtx, num_classes=None):
    # check types
    if not isinstance(cost_mtx, np.ndarray):
        raise TypeError
    if num_classes is None:
        num_classes = cost_mtx.shape[0]
    else:
        num_classes = check_num_classes(num_classes)

    # check cost requirements
    if cost_mtx.shape != (num_classes, num_classes):
        raise ValueError
    if not np.allclose(np.diag(cost_mtx), 0.0):
        raise ValueError
    if np.any(cost_mtx < 0.0, axis=None):
        raise ValueError

    # renormalize for minimum non-zero cost to be unit
    cost_vals = cost_mtx.flatten()
    cost_vals = cost_vals[~np.isclose(cost_vals, 0.0)]
    cost_min = cost_vals.min()
    cost_mtx /= cost_min

    return cost_mtx


def calc_cost(confus_mtx, cost_mtx, average=COST_AVG_DEFAULT):
    # preliminary checks
    confus_mtx = check_confusion_matrix(confus_mtx)
    cost_mtx = check_cost_matrix(cost_mtx)
    if confus_mtx.shape != cost_mtx.shape:
        raise ValueError

    # computations of costs
    cost_all = confus_mtx * cost_mtx  # element-wise multiplication
    cost = np.nan

    if average not in COST_AVERAGE:
        raise NotImplementedError

    elif average == 'micro':
        cost_total = cost_all.sum(axis=None)  # sum all elements
        num_samples = confus_mtx.sum(axis=None)
        cost_avg_total = cost_total / num_samples

        cost = cost_avg_total

    elif average == 'macro':
        cost_class = cost_all.sum(axis=1)  # sum each class separately
        class_supports = confus_mtx.sum(axis=1)
        cost_avg_class = cost_class / class_supports

        cost = cost_avg_class.mean()

    return cost


def calc_cost_base(confus_mtx, cost_mtx, average=COST_AVG_DEFAULT):
    # preliminary checks
    confus_mtx = check_confusion_matrix(confus_mtx)
    cost_mtx = check_cost_matrix(cost_mtx)
    if confus_mtx.shape != cost_mtx.shape:
        raise ValueError

    cost_base = np.nan
    if average not in COST_AVERAGE:
        raise NotImplementedError

    # dummy scenario: predict always the class with lowest average cost
    elif average == 'micro':
        class_supports = confus_mtx.sum(axis=1)
        class_weights = class_supports / class_supports.sum()

        cost_base_class = np.average(cost_mtx, weights=class_weights, axis=0)
        cost_base = cost_base_class.min()

    elif average == 'macro':
        cost_base_class = cost_mtx.mean(axis=0)
        cost_base = cost_base_class.min()

    return cost_base


def calc_cost_max(confus_mtx, cost_mtx, average=COST_AVG_DEFAULT):
    # preliminary checks
    confus_mtx = check_confusion_matrix(confus_mtx)
    cost_mtx = check_cost_matrix(cost_mtx)
    if confus_mtx.shape != cost_mtx.shape:
        raise ValueError

    cost_max = np.nan
    if average not in COST_AVERAGE:
        raise NotImplementedError

    # worst case scenario: predict always the most costly mistake
    elif average == 'micro':
        class_supports = confus_mtx.sum(axis=1)
        class_weights = class_supports / class_supports.sum()

        cost_max_class = cost_mtx.max(axis=1)
        cost_max = np.average(cost_max_class, weights=class_weights)

    elif average == 'macro':
        cost_max_class = cost_mtx.max(axis=1)
        cost_max = cost_max_class.mean()

    return cost_max


def calc_cost_score(confus_mtx, cost_mtx, average=COST_AVG_DEFAULT, reference=COST_REF_DEFAULT):
    cost = calc_cost(confus_mtx, cost_mtx, average=average)

    cost_ref = np.nan
    if reference not in COST_REFERENCE:
        raise NotImplementedError
    elif reference == 'base':
        cost_ref = calc_cost_base(confus_mtx, cost_mtx, average=average)
    elif reference == 'max':
        cost_ref = calc_cost_max(confus_mtx, cost_mtx, average=average)

    # scores convention is the higher the better
    # rescale and normalize:
    # zero (perfect) cost set to 1.0 score
    # reference cost, either dummy baseline (and beyond) or maximum, set to 0.0 score
    score = 1.0 - (cost / cost_ref)
    score = max([score, 0.0])  # retain only better than reference (relevant if baseline)

    return score


def gen_cost_matrix_nominal(num_classes):
    num_classes = check_num_classes(num_classes)

    cost_mtx = np.ones((num_classes, num_classes)) - np.eye(num_classes)
    cost_mtx = check_cost_matrix(cost_mtx)
    return cost_mtx


def gen_cost_matrix_symmetric(num_classes, alpha):
    num_classes = check_num_classes(num_classes)
    if alpha < 0.0:
        raise ValueError

    cost_mtx = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            dist = j - i
            cost = alpha ** (dist - 1)
            cost_mtx[i, j] = cost
            cost_mtx[j, i] = cost
    cost_mtx = check_cost_matrix(cost_mtx)
    return cost_mtx


def gen_cost_matrix_asymmetric(num_classes, alpha, beta, kappa):
    num_classes = check_num_classes(num_classes)
    if (alpha <= 1.0) or (beta <= 1.0) or (kappa <= 0.0):
        raise ValueError

    cost_mtx = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            if i == j:
                continue
            elif i < j:
                dist = j - i
                cost = alpha ** (dist - 1)
                cost_mtx[i, j] = cost
            else:  # i > j
                dist = i - j
                cost = kappa * (alpha ** (dist - 1)) * (beta ** (dist - 1))
                cost_mtx[i, j] = cost
    cost_mtx = check_cost_matrix(cost_mtx)
    return cost_mtx
