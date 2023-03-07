from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin

from sklearn.feature_selection import GenericUnivariateSelect, mutual_info_classif, mutual_info_regression

from Discretizer import DiscretizerX


class FeatSelMIClassif(SelectorMixin, BaseEstimator):
    def __init__(self, hyperparams, random_seq):
        self.hyperparams = hyperparams
        self.random_seq = random_seq

        self.feat_names = None
        self.feat_categ = None

        self.core = None
        self.fitted = False
        self.mask = None

    def set_feature_config(self, feat_names, feat_categ):
        self.feat_names = feat_names
        self.feat_categ = feat_categ

    def fit(self, X, y, **fit_params):
        random_state = self.random_seq.spawn(1)[0].generate_state(1)[0]

        # build
        num_feats = self.hyperparams['num_feats']
        num_neighbs = self.hyperparams['num_neighbs']

        discret_x = DiscretizerX(feat_names=self.feat_names, feat_categ=self.feat_categ, discret_cont=False)
        f_mi_c = lambda XX, yy: mutual_info_classif(discret_x.fit_transform(XX), yy,
                                                    n_neighbors=num_neighbs, discrete_features=self.feat_categ,
                                                    random_state=random_state)

        core = GenericUnivariateSelect(score_func=f_mi_c, mode='k_best', param=num_feats)
        core.fit(X, y)
        support_mask = core.get_support(indices=False)

        self.core = core
        self.fitted = True
        self.mask = support_mask
        return self

    def _get_support_mask(self):
        return self.mask

    def _more_tags(self):
        return {'requires_y': True, 'allow_nan': False}


class FeatSelMIRegress(SelectorMixin, BaseEstimator):
    def __init__(self, hyperparams, random_seq):
        self.hyperparams = hyperparams
        self.random_seq = random_seq

        self.feat_names = None
        self.feat_categ = None

        self.core = None
        self.fitted = False
        self.mask = None

    def set_feature_config(self, feat_names, feat_categ):
        self.feat_names = feat_names
        self.feat_categ = feat_categ

    def fit(self, X, y, **fit_params):
        random_state = self.random_seq.spawn(1)[0].generate_state(1)[0]

        # build
        num_feats = self.hyperparams['num_feats']
        num_neighbs = self.hyperparams['num_neighbs']

        discret_x = DiscretizerX(feat_names=self.feat_names, feat_categ=self.feat_categ, discret_cont=False)
        f_mi_r = lambda X, y: mutual_info_regression(discret_x.fit_transform(X), y,
                                                     n_neighbors=num_neighbs, discrete_features=self.feat_categ,
                                                     random_state=random_state)

        core = GenericUnivariateSelect(score_func=f_mi_r, mode='k_best', param=num_feats)
        core.fit(X, y)
        support_mask = core.get_support(indices=False)

        self.core = core
        self.fitted = True
        self.mask = support_mask
        return self

    def _get_support_mask(self):
        return self.mask

    def _more_tags(self):
        return {'requires_y': True, 'allow_nan': False}
