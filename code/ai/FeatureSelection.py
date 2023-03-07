from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin

from FeatSelMI import FeatSelMIClassif, FeatSelMIRegress
from FeatSelMRMR import FeatSelMRMR
from FeatSelRelief import FeatSelReliefF, FeatSelMultiSURF

from FeatSelEmbed import FeatSelL1Classif, FeatSelL1Regress

WRAP_METHODS = ['linear', 'svm', 'hgb']

from FeatSelRFE import FeatSelRFEClassif, FeatSelRFERegress, FeatSelRFECVClassif, FeatSelRFECVRegress
from FeatSelSeq import FeatSelSeqClassif, FeatSelSeqRegress
from FeatSelGA import FeatSelGAClass, FeatSelGARegr
from FeatSelPSO import FeatSelPSOClass, FeatSelPSORegr

FEATURE_SELECTORS = ['filt-mutual_info_classif', 'filt-mutual_info_regress',
                     'filt-mrmr', 'filt-relieff', 'filt-multisurf',
                     'embed-l1_classif', 'embed-l1_regress',
                     'wrap-rfecv_classif', 'wrap-rfecv_regress', 'wrap-rfe_classif', 'wrap-rfe_regress',
                     'wrap-seq_classif', 'wrap-seq_regress',
                     'wrap-ga_classif', 'wrap-ga_regress', 'wrap-pso_classif', 'wrap-pso_regress']


class FeatureSelector(SelectorMixin, BaseEstimator):
    def __init__(self, method, hyperparams, score_opts, random_seq):
        if method not in FEATURE_SELECTORS:
            raise NotImplementedError
        else:
            self.method = method
            self.hyperparams = hyperparams
            self.score_opts = score_opts
            self.random_seq = random_seq

            self.feat_names = None
            self.feat_categ = None

        if self.method == 'filt-mutual_info_classif':
            selector = FeatSelMIClassif(hyperparams, random_seq)
            self.selector = selector

        elif self.method == 'filt-mutual_info_regress':
            selector = FeatSelMIRegress(hyperparams, random_seq)
            self.selector = selector

        elif self.method == 'filt-mrmr':
            selector = FeatSelMRMR(hyperparams, random_seq)
            self.selector = selector

        elif self.method == 'filt-relieff':
            selector = FeatSelReliefF(hyperparams, random_seq)
            self.selector = selector

        elif self.method == 'filt-multisurf':
            selector = FeatSelMultiSURF(hyperparams, random_seq)
            self.selector = selector

        elif self.method == 'embed-l1_classif':
            selector = FeatSelL1Classif(hyperparams, random_seq)
            self.selector = selector

        elif self.method == 'embed-l1_regress':
            selector = FeatSelL1Regress(hyperparams, random_seq)
            self.selector = selector

        elif self.method == 'wrap-rfe_classif':
            selector = FeatSelRFEClassif(hyperparams, random_seq)
            self.selector = selector

        elif self.method == 'wrap-rfe_regress':
            selector = FeatSelRFERegress(hyperparams, random_seq)
            self.selector = selector

        elif self.method == 'wrap-rfecv_classif':
            selector = FeatSelRFECVClassif(hyperparams, score_opts, random_seq)
            self.selector = selector

        elif self.method == 'wrap-rfecv_regress':
            selector = FeatSelRFECVRegress(hyperparams, score_opts, random_seq)
            self.selector = selector

        elif self.method == 'wrap-seq_classif':
            selector = FeatSelSeqClassif(hyperparams, score_opts, random_seq)
            self.selector = selector

        elif self.method == 'wrap-seq_regress':
            selector = FeatSelSeqRegress(hyperparams, score_opts, random_seq)
            self.selector = selector

        elif self.method == 'wrap-ga_classif':
            selector = FeatSelGAClass(hyperparams, score_opts, random_seq)
            self.selector = selector

        elif self.method == 'wrap-ga_regress':
            selector = FeatSelGARegr(hyperparams, score_opts, random_seq)
            self.selector = selector

        elif self.method == 'wrap-pso_classif':
            selector = FeatSelPSOClass(hyperparams, score_opts, random_seq)
            self.selector = selector

        elif self.method == 'wrap-pso_regress':
            selector = FeatSelPSORegr(hyperparams, score_opts, random_seq)
            self.selector = selector

    def set_feature_config(self, feat_names, feat_categ):
        self.feat_names = feat_names
        self.feat_categ = feat_categ
        self.selector.set_feature_config(feat_names, feat_categ)
        return self

    def get_feature_config(self):
        return self.feat_names, self.feat_categ

    def fit_transform(self, X, y=None, **fit_params):
        return self.selector.fit_transform(X, y, **fit_params)

    def get_support(self, indices=False):
        return self.selector.get_support()

    def inverse_transform(self, X):
        return self.selector.inverse_transform(X)

    def transform(self, X):
        return self.selector.transform(X)

    def _get_support_mask(self):
        return self.selector.get_support(indices=False)

    def _more_tags(self):
        return self.selector._more_tags()
