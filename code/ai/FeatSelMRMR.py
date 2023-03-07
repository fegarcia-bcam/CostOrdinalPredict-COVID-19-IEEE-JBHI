import os
import sys

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin

import pymrmr

from Discretizer import DiscretizerX

import config


class FeatSelMRMR(SelectorMixin, BaseEstimator):
    def __init__(self, hyperparams, random_seq):
        self.hyperparams = hyperparams
        self.random_seq = random_seq

        self.feat_names = None
        self.feat_categ = None

        self.fitted = False
        self.mask = None

    def set_feature_config(self, feat_names, feat_categ):
        self.feat_names = feat_names
        self.feat_categ = feat_categ

    def fit(self, X, y, **fit_params):
        discret_x = DiscretizerX(feat_names=self.feat_names, feat_categ=self.feat_categ, discret_cont=True)
        X_discr = discret_x.fit_transform(X)
        y_discr = y.to_numpy().astype('int')

        df_discr = pd.DataFrame(data=X_discr, columns=self.feat_names)
        df_discr.insert(0, config.VAR_CLASSIF, y_discr)

        method = self.hyperparams['method']
        num_feats = self.hyperparams['num_feats']

        if config.VERBOSE >= 5:
            cols_sel = pymrmr.mRMR(df_discr, method, num_feats)
        else:  # discard PyMRMR's intermediate verbose output
            devnull = open(os.devnull, 'w')
            with _RedirectStdStreams(stdout=devnull, stderr=devnull):
                cols_sel = pymrmr.mRMR(df_discr, method, num_feats)

        cols_all = df_discr.columns.tolist()
        cols_all.remove(config.VAR_CLASSIF)
        support_mask = [col in cols_sel for col in cols_all]
        support_mask = np.asarray(support_mask, dtype=bool)

        self.fitted = True
        self.mask = support_mask
        return self

    def _get_support_mask(self):
        return self.mask

    def _more_tags(self):
        return {'requires_y': True, 'allow_nan': False}


class _RedirectStdStreams(object):
    def __init__(self, *args, **kw):
        sys.stdout.flush()
        self._orig_stdout = sys.stdout
        self._old_stdout_fno = os.dup(sys.stdout.fileno())
        self._devnull = os.open(os.devnull, os.O_WRONLY)

    def __enter__(self):
        self._new_stdout = os.dup(1)
        os.dup2(self._devnull, 1)
        os.close(self._devnull)
        sys.stdout = os.fdopen(self._new_stdout, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._orig_stdout
        sys.stdout.flush()
        os.dup2(self._old_stdout_fno, 1)
