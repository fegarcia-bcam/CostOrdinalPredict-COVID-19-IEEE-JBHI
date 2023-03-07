import numpy as np
import pandas as pd

from imblearn.over_sampling import RandomOverSampler, ADASYN, SMOTE, BorderlineSMOTE
from imblearn.under_sampling import EditedNearestNeighbours, TomekLinks
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids, NearMiss

import config

OVERSAMPLE_METHODS = ['none', 'random', 'adasyn', 'smote', 'smote_border']
UNDERSAMPLE_CLEAN_METHODS = ['none', 'enn', 'tomek']
UNDERSAMPLE_CONTROL_METHODS = ['none', 'random', 'cluster_c', 'near_miss']

SAMPLING_STRATEGY_OVS_MAIN = 'not majority'
SAMPLING_STRATEGY_OVS_AUX = 'minority'
SAMPLING_STRATEGY_UNS_CLEAN = 'all'


def balance(X, y, target_ratios, options, hyperparams, random_seq):
    X_res = X.copy()
    y_res = y.copy()
    # first, discard any classes not relevant to our current (possibly decomposed) model
    X_res, y_res, target_ratios = keep_relevant(X_res, y_res, target_ratios)

    # generate random states
    random_states = random_seq.spawn(1)[0].generate_state(2)
    random_state_ovs, random_state_uns_ctrl = random_states

    # oversample for class balance
    options_ovs = options['balancer']['ovs']
    hyperparams_ovs = hyperparams['balancer']['ovs']
    X_res, y_res, ovs = oversample(X_res, y_res, options_ovs, hyperparams_ovs, random_state_ovs)

    # undersample to clean noisy synthetic samples
    options_uns_clean = options['balancer']['uns_clean']
    hyperparams_uns_clean = hyperparams['balancer']['uns_clean']
    X_res, y_res, uns_clean = undersample_clean(X_res, y_res, options_uns_clean, hyperparams_uns_clean)

    # undersample to control for cost-related target ratios
    options_uns_ctrl = options['balancer']['uns_ctrl']
    hyperparams_uns_ctrl = hyperparams['balancer']['uns_ctrl']
    X_res, y_res, uns_ctrl = undersample_control(X_res, y_res, target_ratios, options_uns_ctrl,
                                                 hyperparams_uns_ctrl, random_state_uns_ctrl)

    balancer = (ovs, uns_clean, uns_ctrl)
    return X_res, y_res, balancer


def keep_relevant(X, y, target_ratios):
    X_relev = X.copy()
    y_relev = y.copy()
    if isinstance(X, pd.DataFrame):
        X_relev = X_relev.to_numpy()
    if isinstance(y, pd.Series):
        y_relev = y_relev.to_numpy()

    if not isinstance(target_ratios, dict):
        target_ratios_relev = target_ratios

    else:
        target_ratios_relev = target_ratios.copy()

        idx_irrelev = pd.Series([False] * y_relev.size)
        for label, ratio in target_ratios.items():
            # mark those samples belonging to classes which will be dropped
            if ~np.isclose(ratio, 0.0):
                continue
            idx_label = (y_relev == label)
            idx_irrelev |= idx_label

            # also remove that class label from the sampling strategy
            target_ratios_relev.pop(label)
        idx_relev = ~idx_irrelev.to_numpy()

        X_relev = X_relev[idx_relev]
        y_relev = y_relev[idx_relev]

    if isinstance(X, pd.DataFrame):
        X_relev = pd.DataFrame(X_relev)
    if isinstance(y, pd.Series):
        y_relev = pd.Series(y_relev)

    return X_relev, y_relev, target_ratios_relev


def oversample(X, y, options, hyperparams, random_state):
    ovs = None
    if options not in OVERSAMPLE_METHODS:
        raise NotImplementedError
    elif options == 'none':  # no oversampling
        pass
    elif options == 'random':
        ovs = RandomOverSampler(sampling_strategy=SAMPLING_STRATEGY_OVS_MAIN, random_state=random_state,
                                **hyperparams)
    elif options == 'adasyn':
        ovs = ADASYN(sampling_strategy=SAMPLING_STRATEGY_OVS_MAIN, random_state=random_state,
                     n_jobs=config.NUM_JOBS_IMBALANCE, **hyperparams)
    elif options == 'smote':
        ovs = SMOTE(sampling_strategy=SAMPLING_STRATEGY_OVS_MAIN, random_state=random_state,
                    n_jobs=config.NUM_JOBS_IMBALANCE, **hyperparams)
    elif options == 'smote_border':
        ovs = BorderlineSMOTE(sampling_strategy=SAMPLING_STRATEGY_OVS_MAIN, random_state=random_state,
                              n_jobs=config.NUM_JOBS_IMBALANCE, **hyperparams)

    if ovs is None:
        X_res = X.copy()
        y_res = y.copy()
    else:
        try:
            X_res, y_res = ovs.fit_resample(X, y)

        except ValueError as ve_main:
            if options == 'adasyn':
                # 'not majority' sampling didn't work well for ADASYN
                # probably be due to a rounding effect, when there is almost balance between majority and (at least one) non-majority class
                # alternatively, let's try 'minority' to focus only in the least represented class
                ovs = ADASYN(sampling_strategy=SAMPLING_STRATEGY_OVS_AUX, random_state=random_state,
                             n_jobs=config.NUM_JOBS_IMBALANCE, **hyperparams)
                try:
                    X_res, y_res = ovs.fit_resample(X, y)

                except ValueError as ve_aux:
                    # 'minority' didn't work either, don't oversample
                    ovs = None
                    X_res = X.copy()
                    y_res = y.copy()

            else:  # unknown cause, should never happen
                raise ve_main

    return X_res, y_res, ovs


def undersample_clean(X, y, options, hyperparams):
    uns_clean = None
    if options not in UNDERSAMPLE_CLEAN_METHODS:
        raise NotImplementedError
    elif options == 'none':  # no cleaning
        pass
    elif options == 'enn':
        uns_clean = EditedNearestNeighbours(sampling_strategy=SAMPLING_STRATEGY_UNS_CLEAN,
                                            n_jobs=config.NUM_JOBS_IMBALANCE, **hyperparams)
    elif options == 'tomek':
        uns_clean = TomekLinks(sampling_strategy=SAMPLING_STRATEGY_UNS_CLEAN,
                               n_jobs=config.NUM_JOBS_IMBALANCE, **hyperparams)

    if uns_clean is None:
        X_res = X.copy()
        y_res = y.copy()
    else:
        X_res, y_res = uns_clean.fit_resample(X, y)
    return X_res, y_res, uns_clean


def undersample_control(X, y, target_ratios, options, hyperparams, random_state):
    sampling_strategy = get_sampling_strategy(y, target_ratios)

    uns_ctrl = None
    if options not in UNDERSAMPLE_CONTROL_METHODS:
        raise NotImplementedError
    elif options == 'none':  # no control
        pass
    elif options == 'random':
        uns_ctrl = RandomUnderSampler(sampling_strategy=sampling_strategy,
                                      random_state=random_state, **hyperparams)  # no support for multiple jobs
    elif options == 'cluster_c':
        uns_ctrl = ClusterCentroids(sampling_strategy=sampling_strategy,
                                    random_state=random_state, **hyperparams)  # no support for multiple jobs
    elif options == 'near_miss':
        uns_ctrl = NearMiss(sampling_strategy=sampling_strategy,
                            n_jobs=config.NUM_JOBS_IMBALANCE, **hyperparams)  # no randomness

    if uns_ctrl is None:
        X_res = X.copy()
        y_res = y.copy()
    else:
        X_res, y_res = uns_ctrl.fit_resample(X, y)
    return X_res, y_res, uns_ctrl


def get_sampling_strategy(y, target_ratios):
    # count number of samples of each class, as present in the dataset
    observ_freqs = []
    for label in target_ratios.keys():
        observ_freq = (y == label).sum()
        observ_freqs.append(observ_freq)
    observ_freqs = np.asarray(observ_freqs)

    # normalized desired ratios of samples with respect to the desired majority class
    relat_ratios = np.asarray(list(target_ratios.values()))
    relat_ratios /= relat_ratios.max(axis=None)

    idx_majority_class = int(np.argmax(relat_ratios))
    target_freq_major = observ_freqs[idx_majority_class]

    # check that the target samples are at most equal to the available
    # if not, decrease step by step until satisfied
    target_freqs = np.zeros_like(relat_ratios)
    while target_freq_major > 0:
        target_freqs = np.round(target_freq_major * relat_ratios)
        if np.all(target_freqs <= observ_freqs):
            break
        else:
            target_freq_major -= 1

    # check if any would get undersampled, we cannot ask as target less samples than observed
    sampl_strateg = {}
    for idx, label in enumerate(target_ratios.keys()):
        sampl_strateg[label] = int(target_freqs[idx])

    return sampl_strateg
