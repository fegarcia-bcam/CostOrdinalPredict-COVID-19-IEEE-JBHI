import itertools
import numpy as np

from scipy.optimize import minimize

PROBA_EPSILON = 1.0e-12
PROBA_BOUNDS = (0.0, 1.0)


# Hastie, Tibshirani 1998 pairwise coupling method
def pairwise_couple(confid_mtx, class_priors):
    # check class_priors, convert format if needed
    if isinstance(class_priors, dict):
        class_priors = np.asarray(list(class_priors.values()))
    if not isinstance(class_priors, np.ndarray):
        raise TypeError
    if np.any(class_priors <= 0.0):
        raise ValueError

    # check confid_mtx matrix
    num_classes = class_priors.size
    confid_mtx = check_confidence_matrix(confid_mtx, num_classes=num_classes)

    # prepare counts matrix
    n = np.full_like(confid_mtx, np.nan)
    class_combinats = itertools.combinations(range(num_classes), 2)
    for i, j in class_combinats:
        nij = class_priors[i] + class_priors[j]
        n[i, j] = nij
        n[j, i] = nij

    # prepare initial guess for optimization
    proba_0 = class_priors / class_priors.sum()  # class frequencies observed during training

    # target
    f_kld = lambda x: kullback_leibler_divergence(x, confid_mtx, n)
    # bounds
    bounds = tuple([PROBA_BOUNDS] * num_classes)
    # constraints
    constraints = ({'type': 'eq', 'fun': lambda p: 1.0 - p.sum()})  # all probabilities must sum to one

    # optimize
    try:
        result = minimize(f_kld, proba_0, bounds=bounds, constraints=constraints)  # algorithm chosen automatically
        if not result.success:
            raise ValueError
        proba = result.x
    # if something fails, use a rescue alternative
    except ValueError:
        proba = pairwise_rescue(confid_mtx)

    return proba


def kullback_leibler_divergence(p, r, n):
    # prepare proba matrix mu
    mu = np.full_like(r, np.nan)
    num_classes, _ = mu.shape
    class_combinats = itertools.combinations(range(num_classes), 2)
    for i, j in class_combinats:
        pij = p[i] + p[j] + PROBA_EPSILON  # fix numerical precision issues
        mu[i, j] = p[i] / pij
        mu[j, i] = p[j] / pij
    with np.errstate(invalid='ignore'):  # disregard warnings due to NaN comparison
        mu[mu < PROBA_EPSILON] = PROBA_EPSILON  # fix numerical precision issues

    # compute Kullback-Leibler divergence, first element-wise
    kl = n * r * np.log2(r / mu)
    kl = np.nansum(kl, axis=None)

    return kl


def pairwise_rescue(confid_mtx):
    confid_mtx = check_confidence_matrix(confid_mtx, num_classes=None)
    num_classes, _ = confid_mtx.shape

    # ill case, when one class clearly dominates and numerical issues prevent convergence
    w_votes = np.nansum(confid_mtx, axis=1)
    idx_winner = int(np.argmax(w_votes))

    proba = np.full((num_classes,), PROBA_EPSILON)
    proba[idx_winner] = 1.0 - (num_classes - 1) * PROBA_EPSILON

    return proba


def check_confidence_matrix(confid_mtx, num_classes=None):
    if num_classes is not None:
        if confid_mtx.shape != (num_classes, num_classes):
            raise ValueError
    else:
        num_classes, _ = confid_mtx.shape
    if not np.isnan(np.diag(confid_mtx)).all():
        raise ValueError

    confid_vec = confid_mtx.flatten()
    if np.count_nonzero(np.isnan(confid_vec)) != num_classes:
        raise ValueError
    confid_vec = confid_vec[~np.isnan(confid_vec)]
    if np.any(confid_vec < 0.0) or np.any(confid_vec > 1.0):
        raise ValueError

    class_combinat = itertools.combinations(range(num_classes), 2)
    for i, j in class_combinat:
        confid_sum = confid_mtx[i, j] + confid_mtx[j, i]
        if not np.isclose(confid_sum, 1.0):
            raise ValueError

    return confid_mtx
