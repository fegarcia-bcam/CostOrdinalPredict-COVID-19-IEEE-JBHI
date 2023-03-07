import numpy as np

from utils_proba import check_proba, bound_proba
from balance import balance

from OrdinalScenario import TAG_NEG, TAG_POS
from OrdinalDeReComp import OrdinalDeReComp, TARGET_RATIOS_POLICIES

import config


class OrdinalDeReCompOrderPart(OrdinalDeReComp):
    def __init__(self, policy,
                 num_classes=config.NUM_CLASSES, labels=config.LABELS, cost_mtx=config.COST_MTX):
        super().__init__(num_classes=num_classes, labels=labels, cost_mtx=cost_mtx)

        self.set_num_decomp()
        self.set_policy(policy)

    def set_num_decomp(self):
        num_classes = self.get_num_classes()
        num_decomp = num_classes - 1
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
        # go through all decompositions, i.e. ordered partitions
        for idx_partit, random_seq in zip(range(num_decomp), random_seqs_decomp):
            # determine the suitable target ratios for this binary decomposed scenario
            target_ratios = self.get_target_ratios(self.get_policy(), idx_partit)
            # call sample balancing routine
            X_bin, y_bin, balancer = balance(X_decomp, y_decomp, target_ratios, options, hyperparams, random_seq)

            # binarize class labels
            labels_neg = labels.iloc[:(idx_partit + 1)]
            labels_pos = labels.iloc[(idx_partit + 1):]
            idx_y_neg = y_bin.isin(labels_neg)
            idx_y_pos = y_bin.isin(labels_pos)
            y_bin[idx_y_neg] = TAG_NEG
            y_bin[idx_y_pos] = TAG_POS

            l_X.append(X_bin)
            l_y.append(y_bin)
            balancers.append(balancer)

        return l_X, l_y, balancers

    # composed multi-class probabilities from binary
    def recompose(self, l_proba_bin):
        num_classes = self.get_num_classes()
        num_decomp = self.get_num_decomp()
        # preliminary checks
        if num_decomp != (num_classes - 1):
            raise RuntimeError
        if not isinstance(l_proba_bin, list):
            raise TypeError
        if len(l_proba_bin) != num_decomp:
            raise ValueError
        for idx_decomp in range(num_decomp):
            proba_bin = l_proba_bin[idx_decomp]
            proba_bin = check_proba(proba_bin, num_classes=2)
            l_proba_bin[idx_decomp] = proba_bin

        num_samples, _ = l_proba_bin[0].shape
        proba_multicl = np.zeros((num_samples, num_classes))

        # go through all decompositions, i.e. ordered partitions
        # estimates by Cardoso et al. 2007 method
        proba_decomp = np.zeros((num_samples, num_decomp))
        for idx_class in range(num_classes):
            if idx_class == 0:  # first class, first decomposition
                proba_curr = l_proba_bin[idx_class][:, TAG_POS]
                proba_decomp[:, idx_class] = proba_curr
                proba_multicl[:, idx_class] = 1.0 - proba_curr

            elif idx_class < num_decomp:  # subsequent classes until the last decomposition
                proba_prev = proba_decomp[:, idx_class - 1]
                proba_curr = l_proba_bin[idx_class][:, TAG_POS]
                proba_decomp[:, idx_class] = proba_curr * proba_prev
                proba_multicl[:, idx_class] = (1.0 - proba_curr) * proba_prev

            else:  # last class
                proba_prev = proba_decomp[:, idx_class - 1]
                proba_multicl[:, idx_class] = proba_prev

        proba_multicl = bound_proba(proba_multicl)
        return proba_multicl

    def get_target_ratios(self, policy, idx_partit):
        labels = self.get_labels()
        target_ratios = dict.fromkeys(labels)

        if policy not in TARGET_RATIOS_POLICIES:
            raise ValueError

        elif policy == 'equal':
            num_classes = self.get_num_classes()
            freq_ratios = {}
            for idx, label in enumerate(labels):
                if idx <= idx_partit:
                    freq_ratios[label] = 1.0 / (idx_partit + 1)
                else:
                    freq_ratios[label] = 1.0 / (num_classes - (idx_partit + 1))
            freq_max = max(list(freq_ratios.values()))

            target_ratios = {label: freq_ratios[label] / freq_max for label in labels}

        elif policy == 'cost':
            # determine relative miss-classification costs
            cost_mtx = self.get_cost_matrix()
            cost_ratios = {}
            for idx, label in enumerate(labels):
                cost_ratio = 1.0
                if idx != idx_partit:
                    cost_false_pos = cost_mtx[idx_partit, idx]
                    cost_false_neg = cost_mtx[idx, idx_partit]
                    cost_ratio = cost_false_neg / cost_false_pos
                cost_ratios[label] = cost_ratio
            target_ratios = cost_ratios

        return target_ratios
