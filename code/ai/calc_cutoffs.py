import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import ecdf, gaussian_kde
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize, differential_evolution, shgo
from scipy.optimize import Bounds, LinearConstraint

import config


def get_distributions(x, y, plot=True):
    # preliminary checks
    if type(x) is pd.Series:
        x = x.to_numpy()
    if type(y) is pd.Series:
        y = y.to_numpy()

    if x.size != y.size:
        raise ValueError

    y_vals = np.unique(y)
    if (y_vals.size != config.NUM_CLASSES) or (set(y_vals) - set(config.LABELS)):
        raise ValueError

    # empirical cumulative distribution functions: step-wise
    l_distrib = {label: dict() for label in config.LABELS}
    for label in config.LABELS:
        idx_sel = (y == label)
        x_sel = x[idx_sel]
        l_distrib[label]['cdf_empir'] = ecdf(x_sel)

    # empirical cumulative distribution functions: smoothed
    # PCHIP 1-D monotonic cubic interpolation
    for label in config.LABELS:
        x_temp = l_distrib[label]['cdf_empir'].cdf.quantiles
        z_temp = l_distrib[label]['cdf_empir'].cdf.probabilities
        l_distrib[label]['min'] = x_temp.min()
        l_distrib[label]['max'] = x_temp.max()
        l_distrib[label]['cdf_interp'] = PchipInterpolator(x_temp, z_temp, extrapolate=False)

    # empirical probability distribution functions: Gaussian Kernel Density estimate
    for label in config.LABELS:
        idx_sel = (y == label)
        x_sel = x[idx_sel]
        l_distrib[label]['pdf_kde'] = gaussian_kde(x_sel)

    # plotting CDFs and PDFs
    if plot:
        fig_cdf, ax_cdf = plt.subplots()
        for label in config.LABELS:
            l_distrib[label]['cdf_empir'].cdf.plot(ax=ax_cdf, color=config.CLASS_COLOURS[label],
                                                   label='Empirical CDF: {}'.format(config.CLASS_NAMES[label]))

        for label in config.LABELS:
            x_tmp = np.linspace(start=l_distrib[label]['min'], stop=l_distrib[label]['max'], endpoint=True, num=100)
            z_tmp = l_distrib[label]['cdf_interp'](x_tmp)
            ax_cdf.plot(x_tmp, z_tmp, color=config.CLASS_COLOURS[label], linestyle='-.',
                        label='Smoothed CDF: {}'.format(config.CLASS_NAMES[label]))

        ax_cdf.autoscale(enable=True, axis='x', tight=True)
        ax_cdf.set_ylim(0.0, 1.0)
        ax_cdf.grid()
        plt.legend()
        plt.show(block=False)

        fig_pdf, ax_pdf = plt.subplots()
        for label in config.LABELS:
            idx_sel = (y == label)
            x_sel = x[idx_sel]
            ax_pdf.hist(x_sel, bins='fd', density=True, color=config.CLASS_COLOURS[label], alpha=0.40,
                        edgecolor='black', linewidth=0.80, label='Histogram: {}'.format(config.CLASS_NAMES[label]))

        for label in config.LABELS:
            x_tmp = np.linspace(start=l_distrib[label]['min'], stop=l_distrib[label]['max'], endpoint=True, num=100)
            z_tmp = l_distrib[label]['pdf_kde'](x_tmp)
            ax_pdf.plot(x_tmp, z_tmp, color=config.CLASS_COLOURS[label], linestyle='-',
                        label='Kernel PDF: {}'.format(config.CLASS_NAMES[label]))

        # for label in config.LABELS:
        #     x_tmp = np.linspace(start=l_distrib[label]['min'], stop=l_distrib[label]['max'], endpoint=True, num=100)
        #     z_tmp = l_distrib[label]['cdf_interp'].derivative()(x_tmp)
        #     ax_pdf.plot(x_tmp, z_tmp, color=config.CLASS_COLOURS[label], linestyle='-.',
        #                  label='Smoothed CDF deriv: {}'.format(config.CLASS_NAMES[label]))

        ax_pdf.autoscale(enable=True, axis='both', tight=True)
        ax_pdf.grid()
        plt.legend()
        plt.show(block=False)

    return l_distrib


# Youden index, generalized to multi-class
# Nakas et al. 2010, Stat Med || doi:10.1002/sim.4044
def youden_generalized(z, distribs, weights=None):
    youden_gen = 0
    for idx_l in range(config.NUM_CLASSES):
        if idx_l == 0:  # first class
            z_temp = z[idx_l]
            distrib_first = distribs[config.LABELS[idx_l]]

            if z_temp < distrib_first['min']:
                cdf_val = 0.0
            elif z_temp > distrib_first['max']:
                cdf_val = 1.0
            else:
                cdf_val = distrib_first['cdf_interp'](z_temp)
            if np.isnan(cdf_val):
                raise RuntimeError

            youden_gen += weights[idx_l] * cdf_val

        elif idx_l == config.NUM_CLASSES - 1:  # last class
            z_temp = z[idx_l - 1]
            distrib_last = distribs[config.LABELS[idx_l]]

            if z_temp < distrib_last['min']:
                cdf_val = 0.0
            elif z_temp > distrib_last['max']:
                cdf_val = 1.0
            else:
                cdf_val = distrib_last['cdf_interp'](z_temp)
            if np.isnan(cdf_val):
                raise RuntimeError

            youden_gen += weights[idx_l] * (1.0 - cdf_val)

        else:  # intermediate classes
            z_pre = z[idx_l - 1]
            z_pos = z[idx_l]
            distrib_curr = distribs[config.LABELS[idx_l]]

            if z_pre < distrib_curr['min']:
                cdf_val_pre = 0.0
            elif z_pre > distrib_curr['max']:
                cdf_val_pre = 1.0
            else:
                cdf_val_pre = distrib_curr['cdf_interp'](z_pre)
            if np.isnan(cdf_val_pre):
                raise RuntimeError

            if z_pos < distrib_curr['min']:
                cdf_val_pos = 0.0
            elif z_pos > distrib_curr['max']:
                cdf_val_pos = 1.0
            else:
                cdf_val_pos = distrib_curr['cdf_interp'](z_pos)
            if np.isnan(cdf_val_pos):
                raise RuntimeError

            youden_gen += weights[idx_l] * (cdf_val_pos - cdf_val_pre)

    return youden_gen


# supported optimization search strategies, used in finding the optimal cut-off values
SUPPORTED_OPTIM_METHODS = ['trust-constr', 'diff_eval', 'shgo']
DEFAULT_OPTIM_METHODS = 'shgo'


# optimal cut-off determination with empirical CDFs
# Nakas et al. 2010, Stat Med || doi:10.1002/sim.4044
def get_optimal_cutoffs(x, y, weights=None, optim_method=DEFAULT_OPTIM_METHODS):
    # preliminary checks
    if weights is None:
        weights = np.ones(shape=(config.NUM_CLASSES,), dtype=float) / config.NUM_CLASSES
    else:
        if weights.size != config.NUM_CLASSES:
            raise ValueError
        if np.any(weights < 0.0):
            raise ValueError
        weights /= np.sum(weights)

    if optim_method not in SUPPORTED_OPTIM_METHODS:
        raise ValueError

    # get empirical cumulative distribution functions
    l_ecdf = get_distributions(x, y, plot=True)

    # bounds and limits
    c_min, c_max = x.min(), x.max()
    c_bounds = Bounds(lb=[c_min] * (config.NUM_CLASSES - 1),
                      ub=[c_max] * (config.NUM_CLASSES - 1))

    c_lim_mtx = np.eye(config.NUM_CLASSES - 1, dtype=int)
    c_lim_mtx[np.triu_indices(n=config.NUM_CLASSES - 1, k=1)] = -1
    if config.NUM_CLASSES > 3:
        c_lim_mtx[np.triu_indices(n=config.NUM_CLASSES - 1, k=2)] = 0
    c_lim_mtx = c_lim_mtx[:-1, :]
    c_limits = LinearConstraint(c_lim_mtx, lb=-np.inf, ub=0.0)

    # initial guess
    l_quants = np.cumsum(weights)
    x_quants = np.quantile(x, q=l_quants)
    c_init = x_quants[:-1]

    # optimization function
    def f_neg_youden(z):
        return -1.0 * youden_generalized(z, distribs=l_ecdf, weights=weights)

    # minimize by different methods supporting bounds & constraints
    if optim_method == 'trust-constr':
        f_optim = minimize(f_neg_youden, x0=c_init,
                           bounds=c_bounds, constraints=[c_limits],
                           method='trust-constr')
    elif optim_method == 'diff_eval':
        f_optim = differential_evolution(f_neg_youden, x0=c_init,
                                         bounds=c_bounds, constraints=[c_limits],
                                         strategy='best1bin', popsize=1000)
    elif optim_method == 'shgo':
        f_optim = shgo(f_neg_youden,
                       bounds=c_bounds, constraints=[c_limits],
                       n=1000)
    else:
        raise RuntimeError

    if not f_optim.success:
        raise RuntimeError

    cutoffs = f_optim.x
    return cutoffs, l_ecdf


def apply_cutoffs(x, cutoffs):
    # preliminary checks
    if type(x) is pd.Series:
        x = x.to_numpy()
    if cutoffs.size != config.NUM_CLASSES - 1:
        raise ValueError

    # start by an empty prediction
    y = np.full_like(x, fill_value=np.nan, dtype=float)

    # check cut-offs in increasing order
    for idx_l in range(config.NUM_CLASSES - 1):
        idx_sel = np.isnan(y) & (x <= cutoffs[idx_l])
        y[idx_sel] = config.LABELS[idx_l]

    # beyond the highest cut-off
    y[np.isnan(y)] = config.LABELS.iloc[-1]
    y = y.astype(int)

    return y


# estimate per-class posterior probabilities, Bayes theorem
def estimate_proba(score_pred, distribs, priors):
    proba_pred = np.zeros(shape=(score_pred.size, config.NUM_CLASSES))
    for idx_score, x in enumerate(score_pred):
        for idx_label, label in enumerate(config.LABELS):
            distrib_temp = distribs[label]
            pdf_val = distrib_temp['pdf_kde'](x)
            proba_pred[idx_score, idx_label] = priors[idx_label] * pdf_val

    bayes_denom = proba_pred.sum(axis=1)
    bayes_denom = np.repeat(bayes_denom[:, np.newaxis], repeats=config.NUM_CLASSES, axis=1)
    proba_pred = np.divide(proba_pred, bayes_denom)
    return proba_pred
