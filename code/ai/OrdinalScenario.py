from utils_format import check_num_classes, check_labels
from utils_cost import check_cost_matrix

import config

TAG_NEG = 0
TAG_POS = 1


class OrdinalScenario:

    num_classes = None
    labels = None
    cost_mtx = None

    def __init__(self, num_classes=config.NUM_CLASSES, labels=config.LABELS, cost_mtx=config.COST_MTX):
        self.set_num_classes(num_classes)
        self.set_labels(labels)
        self.set_cost_matrix(cost_mtx)

    def get_num_classes(self):
        return self.num_classes

    def set_num_classes(self, num_classes):
        num_classes = check_num_classes(num_classes)
        self.num_classes = num_classes
        return self

    def get_labels(self):
        return self.labels

    def set_labels(self, labels):
        labels = check_labels(labels, num_classes=self.get_num_classes())
        self.labels = labels
        return self

    def get_cost_matrix(self):
        return self.cost_mtx

    def set_cost_matrix(self, cost_mtx):
        cost_mtx = check_cost_matrix(cost_mtx, num_classes=self.get_num_classes())
        self.cost_mtx = cost_mtx
        return self
