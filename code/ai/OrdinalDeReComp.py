from OrdinalScenario import OrdinalScenario

import config

ORDINAL_DECOMPOSERS = ['regress', 'one-vs-one', 'ordered-part']
TARGET_RATIOS_POLICIES = ['equal', 'cost']


class OrdinalDeReComp(OrdinalScenario):

    num_decomp = None
    policy = None

    def __init__(self, num_classes=config.NUM_CLASSES, labels=config.LABELS, cost_mtx=config.COST_MTX):
        super().__init__(num_classes=num_classes, labels=labels, cost_mtx=cost_mtx)

    def get_num_decomp(self):
        return self.num_decomp

    def set_num_decomp(self):
        raise NotImplementedError

    def get_policy(self):
        return self.policy

    def set_policy(self, policy):
        raise NotImplementedError

    def decompose(self, X, y, options, hyperparams, random_seq):
        raise NotImplementedError

    def recompose(self, l_proba_bin):
        raise NotImplementedError
