import numpy as np
import pandas as pd

from utils_format import check_num_classes, check_labels, NUM_DIMS_MATRIX
from utils_cost import check_cost_matrix

PROBA_DECISION = ['map', 'bayes']
DEFAULT_PROBA_DECIS = 'map'

PROBA_LOWER_LIMIT = 1.0e-4


def check_proba(proba, num_classes=None):
    # check type
    if not isinstance(proba, (np.ndarray, pd.DataFrame)):
        raise TypeError

    # check dimensions
    if proba.ndim != NUM_DIMS_MATRIX:
        raise ValueError
    if num_classes is not None:
        num_classes = check_num_classes(num_classes)
        _, num_proba_cols = proba.shape
        if num_proba_cols != num_classes:
            raise ValueError

    # convert type to numpy
    if isinstance(proba, pd.DataFrame):
        proba = proba.to_numpy()

    # check values
    if np.any(proba < 0.0) or np.any(proba > 1.0):
        raise ValueError

    proba_sum = proba.sum(axis=1)
    if not np.isclose(proba_sum, 1.0).all():
        raise ValueError

    return proba


def bound_proba(proba, num_classes=None):
    proba = check_proba(proba, num_classes=num_classes)

    # find and compensate too small probabilities
    idx_proba_low = (proba < PROBA_LOWER_LIMIT)
    proba[idx_proba_low] = PROBA_LOWER_LIMIT

    # rescale for each row to sum 1
    proba_sum = proba.sum(axis=1, keepdims=True)
    proba /= proba_sum

    return proba


def classify_proba(proba_pred, labels, cost_mtx, decision=DEFAULT_PROBA_DECIS, num_classes=None):
    if decision not in PROBA_DECISION:
        raise ValueError

    if decision == 'map':
        classes_pred, proba_pred = classify_proba_map(proba_pred, labels, num_classes=num_classes)
    elif decision == 'bayes':
        classes_pred, proba_pred = classify_proba_bayes(proba_pred, labels, cost_mtx, num_classes=num_classes)
    else:
        raise ValueError

    return classes_pred, proba_pred


def classify_proba_map(proba_pred, labels, num_classes=None):
    proba_pred = check_proba(proba_pred, num_classes=num_classes)
    labels = check_labels(labels, num_classes=num_classes)

    # classify by Maximum A Posteriori (MAP)
    idx_class_pred = proba_pred.argmax(axis=1)
    classes_pred = [labels.iloc[idx] for idx in idx_class_pred]
    classes_pred = pd.Series(classes_pred)

    return classes_pred, proba_pred


def classify_proba_bayes(proba_pred, labels, cost_mtx, num_classes=None):
    proba_pred = check_proba(proba_pred, num_classes=num_classes)
    labels = check_labels(labels, num_classes=num_classes)
    cost_mtx = check_cost_matrix(cost_mtx, num_classes=num_classes)
    if num_classes is None:
        num_classes = proba_pred.shape[-1]

    # classify by minimum Bayes risk
    risk_pred = np.zeros_like(proba_pred)
    for n in range(num_classes):
        risks = cost_mtx[:, n]
        risk_pred[:, n] = proba_pred.dot(risks)

    idx_class_pred = risk_pred.argmin(axis=1)
    classes_pred = [labels.iloc[idx] for idx in idx_class_pred]
    classes_pred = pd.Series(classes_pred)

    return classes_pred, proba_pred
