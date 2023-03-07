from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin

from sklearn.feature_selection import RFE, RFECV

from sklearn.linear_model import LogisticRegression, Ridge

from sklearn.model_selection import StratifiedKFold

from imblearn.over_sampling import RandomOverSampler

from ImbalancedPipeline import ImbalancedPipeline

from Discretizer import DiscretizerY

from calc_scoring import calc_score

import config


class FeatSelRFEClassif(SelectorMixin, BaseEstimator):
    def __init__(self, hyperparams, random_seq):
        self.hyperparams = hyperparams
        self.random_seq = random_seq

        self.feat_names = None
        self.feat_categ = None

        self.fitted = False
        self.mask = None

        random_st_bal, random_st_clf, random_st_cv = self.random_seq.spawn(1)[0].generate_state(3)

        # build
        balancer = RandomOverSampler(sampling_strategy='not majority', shrinkage=None,
                                     random_state=random_st_bal)

        C = self.hyperparams['C']
        classifier = LogisticRegression(C=C, penalty='l2', multi_class='multinomial',
                                        solver='saga', max_iter=int(1e4), random_state=random_st_clf)

        estimator = ImbalancedPipeline([('balancer', balancer), ('classifier', classifier)])

        importance_getter = 'named_steps.classifier.coef_'

        step = self.hyperparams['step']
        num_feats = self.hyperparams['num_feats']
        core = RFE(estimator=estimator, n_features_to_select=num_feats,
                   step=step, importance_getter=importance_getter)
        self.core = core

    def set_feature_config(self, feat_names, feat_categ):
        self.feat_names = feat_names
        self.feat_categ = feat_categ

    def fit(self, X, y, **fit_params):
        self.core.fit(X, y, **fit_params)

        self.fitted = True
        self.mask = self.core.get_support(indices=False)
        return self

    def _get_support_mask(self):
        return self.mask

    def _more_tags(self):
        return {'requires_y': True, 'allow_nan': False}


class FeatSelRFERegress(SelectorMixin, BaseEstimator):
    def __init__(self, hyperparams, random_seq):
        self.hyperparams = hyperparams
        self.random_seq = random_seq

        self.feat_names = None
        self.feat_categ = None

        self.fitted = False
        self.mask = None

        random_st_bal, random_st_reg, random_st_cv = self.random_seq.spawn(1)[0].generate_state(3)

        # build
        balancer = RandomOverSampler(sampling_strategy='not majority', shrinkage=None,
                                     random_state=random_st_bal)

        alpha = self.hyperparams['alpha']
        regressor = Ridge(alpha=alpha,
                          solver='saga', max_iter=int(1e4), random_state=random_st_reg)

        estimator = ImbalancedPipeline([('balancer', balancer), ('regressor', regressor)])

        importance_getter = 'named_steps.regressor.coef_'

        step = self.hyperparams['step']
        num_feats = self.hyperparams['num_feats']
        core = RFE(estimator=estimator, n_features_to_select=num_feats,
                   step=step, importance_getter=importance_getter)
        self.core = core

    def set_feature_config(self, feat_names, feat_categ):
        self.feat_names = feat_names
        self.feat_categ = feat_categ

    def fit(self, X, y, **fit_params):
        self.core.fit(X, y, **fit_params)

        self.fitted = True
        self.mask = self.core.get_support(indices=False)
        return self

    def _get_support_mask(self):
        return self.mask

    def _more_tags(self):
        return {'requires_y': True, 'allow_nan': False}


class FeatSelRFECVClassif(SelectorMixin, BaseEstimator):
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
        balancer = RandomOverSampler(sampling_strategy='not majority', shrinkage=None,
                                     random_state=random_st_bal)

        C = self.hyperparams['C']
        classifier = LogisticRegression(C=C, penalty='l2', multi_class='multinomial',
                                        solver='saga', max_iter=int(1e4), random_state=random_st_clf)

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

        num_cv_folds = self.hyperparams['num_cv_folds']
        cv = StratifiedKFold(n_splits=num_cv_folds, shuffle=True, random_state=random_st_cv)

        importance_getter = 'named_steps.classifier.coef_'

        step = self.hyperparams['step']
        min_feats = self.hyperparams['min_feats']
        core = RFECV(estimator=estimator, step=step, min_features_to_select=min_feats,
                     scoring=f_scoring_cv, cv=cv, importance_getter=importance_getter,
                     n_jobs=config.NUM_JOBS_RFECV)
        self.core = core

    def set_feature_config(self, feat_names, feat_categ):
        self.feat_names = feat_names
        self.feat_categ = feat_categ

    def fit(self, X, y, **fit_params):
        self.core.fit(X, y, **fit_params)

        self.fitted = True
        self.mask = self.core.get_support(indices=False)
        return self

    def _get_support_mask(self):
        return self.mask

    def _more_tags(self):
        return {'requires_y': True, 'allow_nan': False}


class FeatSelRFECVRegress(SelectorMixin, BaseEstimator):
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
        balancer = RandomOverSampler(sampling_strategy='not majority', shrinkage=None,
                                     random_state=random_st_bal)

        alpha = self.hyperparams['alpha']
        regressor = Ridge(alpha=alpha,
                          solver='saga', max_iter=int(1e4), random_state=random_st_reg)

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

        num_cv_folds = self.hyperparams['num_cv_folds']
        cv = StratifiedKFold(n_splits=num_cv_folds, shuffle=True, random_state=random_st_cv)

        importance_getter = 'named_steps.regressor.coef_'

        step = self.hyperparams['step']
        min_feats = self.hyperparams['min_feats']
        core = RFECV(estimator=estimator, step=step, min_features_to_select=min_feats,
                     scoring=f_scoring_cv, cv=cv, importance_getter=importance_getter,
                     n_jobs=config.NUM_JOBS_RFECV)
        self.core = core

    def set_feature_config(self, feat_names, feat_categ):
        self.feat_names = feat_names
        self.feat_categ = feat_categ

    def fit(self, X, y, **fit_params):
        self.core.fit(X, y, **fit_params)

        self.fitted = True
        self.mask = self.core.get_support(indices=False)
        return self

    def _get_support_mask(self):
        return self.mask

    def _more_tags(self):
        return {'requires_y': True, 'allow_nan': False}
