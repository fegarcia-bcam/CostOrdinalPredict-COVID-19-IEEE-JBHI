import copy

from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin

from sklearn.feature_selection import SequentialFeatureSelector

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC, SVR

from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from sklearn.model_selection import StratifiedKFold

from imblearn.over_sampling import RandomOverSampler

from ImbalancedPipeline import ImbalancedPipeline

from Discretizer import DiscretizerY

from calc_scoring import calc_score

from FeatureSelection import WRAP_METHODS

import config


class FeatSelSeqClassif(SelectorMixin, BaseEstimator):
    def __init__(self, hyperparams, score_opts, random_seq):
        self.hyperparams = hyperparams
        self.score_opts = score_opts
        self.random_seq = random_seq

        self.feat_names = None
        self.feat_categ = None

        self.fitted = False
        self.mask = None

        random_st_bal, random_st_clf, random_st_cv = self.random_seq.spawn(1)[0].generate_state(3)

        # build
        num_feats = self.hyperparams['num_feats']
        direction = self.hyperparams['direction']
        num_cv_folds = self.hyperparams['num_cv_folds']

        hyperparams_wrap = copy.deepcopy(self.hyperparams)
        hyperparams_remove = ['num_feats', 'direction', 'num_cv_folds']
        for hyp in hyperparams_remove:
            hyperparams_wrap.pop(hyp, None)
        method_wrap = hyperparams_wrap.pop('method', None)
        if method_wrap is None:
            raise RuntimeError
        elif method_wrap not in WRAP_METHODS:
            raise NotImplementedError

        balancer = RandomOverSampler(sampling_strategy='not majority', shrinkage=None,
                                     random_state=random_st_bal)
        classifier = None
        if method_wrap == 'linear':
            classifier = LogisticRegression(**hyperparams_wrap, random_state=random_st_clf)
        elif method_wrap == 'svm':
            classifier = SVC(**hyperparams_wrap, random_state=random_st_clf)
        elif method_wrap == 'hgb':
            classifier = HistGradientBoostingClassifier(**hyperparams_wrap, random_state=random_st_clf)
        if classifier is None:
            raise RuntimeError

        estimator = ImbalancedPipeline([('balancer', balancer), ('classifier', classifier)])

        def f_scoring(est, XX, yy, score_opts):
            labels = config.LABELS
            cost_mtx = config.COST_MTX

            scorer_name = score_opts['scorer_name']
            scorer_args = score_opts['scorer_args']

            y_true = yy
            y_pred = est.predict(XX)
            proba_pred = est.predict_proba(XX)

            score = calc_score(scorer_name, y_true, y_pred, proba_pred,
                               labels=labels, cost_mtx=cost_mtx, extra_args=scorer_args)
            return score

        f_scoring_cv = lambda est, X, y: f_scoring(est, X, y, self.score_opts)
        cv = StratifiedKFold(n_splits=num_cv_folds, shuffle=True, random_state=random_st_cv)

        core = SequentialFeatureSelector(estimator=estimator, n_features_to_select=num_feats,
                                         direction=direction, scoring=f_scoring_cv, cv=cv,
                                         n_jobs=config.NUM_JOBS_SEQ_FEAT_SEL)
        self.core = core

    def set_feature_config(self, feat_names, feat_categ):
        self.feat_names = feat_names
        self.feat_categ = feat_categ

    def fit(self, X, y, **fit_params):
        self.core.fit(X, y)

        self.fitted = True
        self.mask = self.core.get_support(indices=False)
        return self

    def _get_support_mask(self):
        return self.mask

    def _more_tags(self):
        allow_nan = (self.hyperparams['method'] == 'hgb')
        return {'requires_y': True, 'allow_nan': allow_nan}


class FeatSelSeqRegress(SelectorMixin, BaseEstimator):
    def __init__(self, hyperparams, score_opts, random_seq):
        self.hyperparams = hyperparams
        self.score_opts = score_opts
        self.random_seq = random_seq

        self.feat_names = None
        self.feat_categ = None

        self.fitted = False
        self.mask = None

        random_st_bal, random_st_reg, random_st_cv = self.random_seq.spawn(1)[0].generate_state(3)

        # build
        num_feats = self.hyperparams['num_feats']
        direction = self.hyperparams['direction']
        num_cv_folds = self.hyperparams['num_cv_folds']

        hyperparams_wrap_r = copy.deepcopy(self.hyperparams)
        hyperparams_remove = ['num_feats', 'direction', 'num_cv_folds']
        for hyp in hyperparams_remove:
            hyperparams_wrap_r.pop(hyp, None)
        method_wrap = hyperparams_wrap_r.pop('method', None)
        if method_wrap is None:
            raise RuntimeError
        elif method_wrap not in WRAP_METHODS:
            raise NotImplementedError

        balancer = RandomOverSampler(sampling_strategy='not majority', shrinkage=None,
                                     random_state=random_st_bal)
        regressor = None
        if method_wrap == 'linear':
            regressor = Ridge(**hyperparams_wrap_r, random_state=random_st_reg)
        elif method_wrap == 'svm':
            regressor = SVR(**hyperparams_wrap_r)
        elif method_wrap == 'hgb':
            regressor = HistGradientBoostingRegressor(**hyperparams_wrap_r, random_state=random_st_reg)
        if regressor is None:
            raise RuntimeError

        estimator = ImbalancedPipeline([('balancer', balancer), ('regressor', regressor)])

        def f_scoring(est, XX, yy, score_opts):
            labels = config.LABELS
            cost_mtx = config.COST_MTX

            scorer_name = score_opts['scorer_name']
            scorer_args = score_opts['scorer_args']

            y_true = yy
            y_pred = est.predict(XX)
            proba_pred = None

            discret_y = DiscretizerY(num_classes=config.NUM_CLASSES)
            y_pred = discret_y.transform(y_pred)

            score = calc_score(scorer_name, y_true, y_pred, proba_pred,
                               labels=labels, cost_mtx=cost_mtx, extra_args=scorer_args)
            return score

        f_scoring_cv = lambda est, X, y: f_scoring(est, X, y, self.score_opts)
        cv = StratifiedKFold(n_splits=num_cv_folds, shuffle=True, random_state=random_st_cv)

        core = SequentialFeatureSelector(estimator=estimator, n_features_to_select=num_feats,
                                         direction=direction, scoring=f_scoring_cv, cv=cv,
                                         n_jobs=config.NUM_JOBS_SEQ_FEAT_SEL)
        self.core = core

    def set_feature_config(self, feat_names, feat_categ):
        self.feat_names = feat_names
        self.feat_categ = feat_categ

    def fit(self, X, y, **fit_params):
        self.core.fit(X, y)

        self.fitted = True
        self.mask = self.core.get_support(indices=False)
        return self

    def _get_support_mask(self):
        return self.mask

    def _more_tags(self):
        allow_nan = (self.hyperparams['method'] == 'hgb')
        return {'requires_y': True, 'allow_nan': allow_nan}
