import warnings

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin

from skrebate import ReliefF, MultiSURF

from Discretizer import DiscretizerX


class _ReliefF(ReliefF):
    def _more_tags(self):
        return {'allow_nan': True}


class FeatSelReliefF(SelectorMixin, BaseEstimator):
    def __init__(self, hyperparams, random_seq):
        self.hyperparams = hyperparams
        self.random_seq = random_seq

        self.feat_names = None
        self.feat_categ = None

        self.fitted = False
        self.mask = None

        # build
        num_feats = self.hyperparams['num_feats']
        num_neighbs = self.hyperparams['num_neighbs']
        discr_thresh = self.hyperparams['discr_thresh']

        core = _ReliefF(n_features_to_select=num_feats, n_neighbors=num_neighbs, discrete_threshold=discr_thresh)
        self.core = core

    def set_feature_config(self, feat_names, feat_categ):
        self.feat_names = feat_names
        self.feat_categ = feat_categ

    def fit(self, X, y, **fit_params):
        discret_x = DiscretizerX(feat_names=self.feat_names, feat_categ=self.feat_categ, discret_cont=False)
        try:
            X = discret_x.fit_transform(X)
        except ValueError:
            # discretizer fails to handle NaNs, which exist if there was no imputation
            # but without imputation, discretization is not necessary
            pass

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
            self.core.fit(X, y)

        num_dims = X.shape[1]
        idx = self.core.top_features_[:self.core.n_features_to_select]
        support_mask = np.zeros(num_dims, dtype=bool)
        support_mask[idx] = True

        self.fitted = True
        self.mask = support_mask
        return self

    def _get_support_mask(self):
        return self.mask

    def _more_tags(self):
        return {'requires_y': True, 'allow_nan': True}


class _MultiSURF(MultiSURF):
    def _more_tags(self):
        return {'allow_nan': True}


class FeatSelMultiSURF(SelectorMixin, BaseEstimator):
    def __init__(self, hyperparams, random_seq):
        self.hyperparams = hyperparams
        self.random_seq = random_seq

        self.feat_names = None
        self.feat_categ = None

        self.fitted = False
        self.mask = None

        # build
        num_feats = self.hyperparams['num_feats']
        discr_thresh = self.hyperparams['discr_thresh']

        core = _MultiSURF(n_features_to_select=num_feats, discrete_threshold=discr_thresh)
        self.core = core

    def set_feature_config(self, feat_names, feat_categ):
        self.feat_names = feat_names
        self.feat_categ = feat_categ

    def fit(self, X, y, **fit_params):
        discret_x = DiscretizerX(feat_names=self.feat_names, feat_categ=self.feat_categ, discret_cont=False)
        try:
            X = discret_x.fit_transform(X)
        except ValueError:
            # discretizer fails to handle NaNs, which exist if there was no imputation
            # but without imputation, discretization is not necessary
            pass

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
            self.core.fit(X, y)

        num_dims = X.shape[1]
        idx = self.core.top_features_[:self.core.n_features_to_select]
        support_mask = np.zeros(num_dims, dtype=bool)
        support_mask[idx] = True

        self.fitted = True
        self.mask = support_mask
        return self

    def _get_support_mask(self):
        return self.mask

    def _more_tags(self):
        return {'requires_y': True, 'allow_nan': True}
