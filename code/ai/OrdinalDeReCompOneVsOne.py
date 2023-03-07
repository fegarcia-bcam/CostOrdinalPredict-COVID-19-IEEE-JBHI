import itertools
import numpy as np

from utils_proba import check_proba, bound_proba
from balance import balance
from pairwise_couple import pairwise_couple

from OrdinalScenario import TAG_NEG, TAG_POS
from OrdinalDeReComp import OrdinalDeReComp, TARGET_RATIOS_POLICIES

import config


class OrdinalDeReCompOneVsOne(OrdinalDeReComp):
    def __init__(self, policy,
                 num_classes=config.NUM_CLASSES, labels=config.LABELS, cost_mtx=config.COST_MTX):
        super().__init__(num_classes=num_classes, labels=labels, cost_mtx=cost_mtx)

        self.set_num_decomp()
        self.set_policy(policy)

    def set_num_decomp(self):
        num_classes = self.get_num_classes()
        num_decomp = (num_classes * (num_classes - 1)) // 2
        self.num_decomp = num_decomp
        return self

    def set_policy(self, policy):
        if policy not in TARGET_RATIOS_POLICIES:
            raise ValueError
        self.policy = policy
        return self

    def decompose(self, X, y, options, hyperparams, random_seq):
        labels = self.get_labels()
        num_classes = self.get_num_classes()

        X_decomp = X.copy()
        y_decomp = y.copy()

        num_decomp = self.get_num_decomp()
        # generate random sequences
        random_seqs_decomp = random_seq.spawn(num_decomp)

        l_X = []
        l_y = []
        balancers = []
        # go through all decompositions, i.e. combinations of pairs of classes
        class_combinats = itertools.combinations(range(num_classes), 2)
        for (idx_class_neg, idx_class_pos), random_seq in zip(class_combinats, random_seqs_decomp):
            # determine the suitable target ratios for this binary decomposed scenario
            target_ratios = self.get_target_ratios(self.get_policy(), idx_class_neg, idx_class_pos)
            # call sample balancing routine
            X_bin, y_bin, balancer = balance(X_decomp, y_decomp, target_ratios, options, hyperparams, random_seq)

            # binarize class labels
            label_neg = labels.iloc[idx_class_neg]
            label_pos = labels.iloc[idx_class_pos]
            idx_y_neg = (y_bin == label_neg)
            idx_y_pos = (y_bin == label_pos)
            y_bin[idx_y_neg] = TAG_NEG
            y_bin[idx_y_pos] = TAG_POS

            l_X.append(X_bin)
            l_y.append(y_bin)
            balancers.append(balancer)

        return l_X, l_y, balancers

    # composed multi-class probabilities from binary
    def recompose(self, l_proba_bin):
        labels = self.get_labels()
        num_classes = self.get_num_classes()
        num_decomp = self.get_num_decomp()

        # preliminary checks
        if not isinstance(l_proba_bin, list):
            raise TypeError
        if len(l_proba_bin) != num_decomp:
            raise ValueError
        for idx_decomp in range(num_decomp):
            proba_bin = l_proba_bin[idx_decomp]
            proba_bin = check_proba(proba_bin, num_classes=2)
            l_proba_bin[idx_decomp] = proba_bin

        num_samples, _ = l_proba_bin[0].shape
        class_priors = {label: 1.0 / num_classes for label in labels}

        # go through all combinations of pairs of classes
        # estimates by Hastie, Tibshirani 1998 pairwise coupling method
        confid_mtx = np.full((num_samples, num_classes, num_classes), np.nan)
        class_combinats = itertools.combinations(range(num_classes), 2)
        for idx_decomp, (idx_class_neg, idx_class_pos) in enumerate(class_combinats):
            proba_bin = l_proba_bin[idx_decomp]
            proba_neg = proba_bin[:, TAG_NEG]
            proba_pos = proba_bin[:, TAG_POS]
            confid_mtx[:, idx_class_neg, idx_class_pos] = proba_neg
            confid_mtx[:, idx_class_pos, idx_class_neg] = proba_pos

        proba_multicl = [pairwise_couple(confid_mtx[i, :, :], class_priors) for i in range(num_samples)]
        proba_multicl = np.stack(proba_multicl, axis=0)
        proba_multicl = bound_proba(proba_multicl)

        return proba_multicl

    def get_target_ratios(self, policy, idx_class_neg, idx_class_pos):
        labels = self.get_labels()
        label_neg = labels.iloc[idx_class_neg]
        label_pos = labels.iloc[idx_class_pos]

        target_ratios = {}
        if policy not in TARGET_RATIOS_POLICIES:
            raise ValueError

        elif policy == 'equal':
            target_ratios = {label: 0 for label in labels}
            target_ratios[label_neg] = 1.0
            target_ratios[label_pos] = 1.0

        elif policy == 'cost':
            # determine relative miss-classification costs
            cost_mtx = self.get_cost_matrix()
            cost_false_pos = cost_mtx[idx_class_neg, idx_class_pos]
            cost_false_neg = cost_mtx[idx_class_pos, idx_class_neg]
            cost_ratio = cost_false_neg / cost_false_pos

            target_ratios = {label: 0 for label in labels}
            target_ratios[label_neg] = 1.0
            target_ratios[label_pos] = cost_ratio

        return target_ratios
