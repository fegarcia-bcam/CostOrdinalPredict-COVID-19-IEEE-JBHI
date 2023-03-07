import numpy as np

from utils_format import check_values
from utils_proba import bound_proba
from balance import balance

from OrdinalDeReComp import OrdinalDeReComp

import config

PROBA_EPSILON = 1.0e-12


class OrdinalDeReCompRegress(OrdinalDeReComp):
    def __init__(self, policy,
                 num_classes=config.NUM_CLASSES, labels=config.LABELS, cost_mtx=config.COST_MTX):
        super().__init__(num_classes=num_classes, labels=labels, cost_mtx=cost_mtx)

        self.set_num_decomp()
        self.set_policy(policy)

    def set_num_decomp(self):
        num_decomp = 1
        self.num_decomp = num_decomp
        return self

    def set_policy(self, policy):
        if policy != 'equal':
            raise ValueError
        self.policy = policy
        return self

    def decompose(self, X, y, options, hyperparams, random_seq):
        labels = self.get_labels()
        num_classes = self.get_num_classes()

        X_decomp = X.copy()
        y_decomp = y.copy()
        # go through all classes and convert labels into indices/values
        for idx_class in range(num_classes):
            label = labels.iloc[idx_class]
            idx_y_class = (y == label)
            y_decomp[idx_y_class] = idx_class
        y_decomp = check_values(y_decomp, num_values=num_classes)

        # generate random sequence
        random_seqs_decomp = random_seq.spawn(1)[0]

        # call sample balancing routine
        target_ratios = self.get_target_ratios()
        X_decomp, y_decomp, balancer = balance(X_decomp, y_decomp, target_ratios, options, hyperparams, random_seqs_decomp)

        l_X = [X_decomp]
        l_y = [y_decomp]
        balancers = [balancer]

        return l_X, l_y, balancers

    def recompose(self, values_pred):
        num_pred = values_pred.size
        num_classes = self.get_num_classes()
        values_ref = np.arange(num_classes)

        # prepare matrices to compare absolute value differences (i.e. distances)
        vec_pred = values_pred.reshape((num_pred, -1))  # column vector
        vec_ref = values_ref.reshape((-1, num_classes))  # row vector
        mtx_pred = np.tile(vec_pred, (1, num_classes))  # matrix, repeat by columns
        mtx_ref = np.tile(vec_ref, (num_pred, 1))  # matrix, repeat by rows
        mtx_dist = np.abs(mtx_pred - mtx_ref)

        # transform distances in approximate probabilities by taking the inverse
        proba_multicl = 1.0 / (mtx_dist + PROBA_EPSILON)
        proba_sum = proba_multicl.sum(axis=1, keepdims=True)
        proba_multicl /= proba_sum

        proba_multicl = bound_proba(proba_multicl)

        return proba_multicl

    def get_target_ratios(self):
        labels = self.get_labels()
        target_ratios = {label: 1.0 for label in labels}

        return target_ratios
