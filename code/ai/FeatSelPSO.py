import copy

from abc import abstractmethod

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC, SVR

from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from sklearn.model_selection import StratifiedKFold, cross_val_predict

from imblearn.over_sampling import RandomOverSampler

from pyswarms.discrete import BinaryPSO

from Discretizer import DiscretizerY
from ImbalancedPipeline import ImbalancedPipeline

from calc_scoring import calc_score

from FeatureSelection import WRAP_METHODS

import config


class FeatSelPSO(SelectorMixin, BaseEstimator):
    def __init__(self, hyperparams, score_opts, random_seq):
        self.hyperparams = hyperparams
        self.score_opts = score_opts
        self.random_seq = random_seq

        self.feat_names = None
        self.feat_categ = None

        self.fitted = False
        self.mask = None

        # collect main configuration info
        hyperparams_wrap = copy.deepcopy(self.hyperparams)
        hyperparams_remove = ['num_iters', 'num_parts', 'options', 'vel_clamp']
        for hyp in hyperparams_remove:
            hyperparams_wrap.pop(hyp, None)

        method_wrap = hyperparams_wrap.pop('method', None)
        if method_wrap is None:
            raise RuntimeError
        elif method_wrap not in WRAP_METHODS:
            raise NotImplementedError

        penaliz = hyperparams_wrap.pop('penaliz', None)
        if penaliz is None:
            raise RuntimeError

        num_cv_folds = hyperparams_wrap.pop('num_cv_folds', None)
        if num_cv_folds is None:
            raise RuntimeError

        support_categ_feats = hyperparams_wrap.pop('support_categ_feats', False)

        self.method_wrap = method_wrap
        self.hyperparams_wrap = hyperparams_wrap
        self.support_categ_feats = support_categ_feats
        self.penaliz = penaliz
        self.num_cv_folds = num_cv_folds

        # build optimizer
        num_iters = self.hyperparams['num_iters']
        num_parts = self.hyperparams['num_parts']
        options = self.hyperparams['options']
        vel_clamp = self.hyperparams['vel_clamp']

        optimizer = _PSOptimizer(num_iters=num_iters, num_parts=num_parts, options=options, vel_clamp=vel_clamp)
        self.optimizer = optimizer

    def set_feature_config(self, feat_names, feat_categ):
        self.feat_names = feat_names
        self.feat_categ = feat_categ

    @abstractmethod
    def fit(self, X, y, **fit_params):
        return self

    def _get_support_mask(self):
        return self.mask

    def _more_tags(self):
        allow_nan = (self.hyperparams['method'] == 'hgb')
        return {'requires_y': True, 'allow_nan': allow_nan}


class FeatSelPSOClass(FeatSelPSO):
    def __init__(self, hyperparams, score_opts, random_seq):
        super(FeatSelPSOClass, self).__init__(hyperparams, score_opts, random_seq)

    def fit(self, X, y, **fit_params):
        # assemble all elements necessary to create the specific loss function
        random_st_bal, random_st_clf, random_st_cv = self.random_seq.spawn(1)[0].generate_state(3)
        balancer = RandomOverSampler(sampling_strategy='not majority', shrinkage=None,
                                     random_state=random_st_bal)
        classifier = None
        if self.method_wrap == 'linear':
            classifier = LogisticRegression(**self.hyperparams_wrap, random_state=random_st_clf)
        elif self.method_wrap == 'svm':
            classifier = SVC(**self.hyperparams_wrap, random_state=random_st_clf)
        elif self.method_wrap == 'hgb':
            classifier = HistGradientBoostingClassifier(**self.hyperparams_wrap, random_state=random_st_clf)
        if classifier is None:
            raise RuntimeError
        estimator = ImbalancedPipeline([('balancer', balancer), ('classifier', classifier)])

        # define loss
        def f_loss_particle(particle, method_wrap, support_categ_feats, num_cv_folds, penaliz, score_opts):
            part_mask = np.asarray(particle, dtype=bool)
            X_sel = X[:, part_mask]
            num_dims_orig = X.shape[1]
            num_dims_sel = X_sel.shape[1]

            if method_wrap == 'hgb' and support_categ_feats:
                feats_categ_aux = np.asarray(self.feat_categ, dtype=bool)
                feats_categ_aux = feats_categ_aux[part_mask]
                feats_categ_aux = feats_categ_aux.tolist()

                estimator.named_steps['classifier'].set_params(categorical_features=feats_categ_aux)

            cv = StratifiedKFold(n_splits=num_cv_folds, shuffle=True, random_state=random_st_cv)

            def f_scoring(est, XX, yy):
                scorer_name = score_opts['scorer_name']
                scorer_args = score_opts['scorer_args']

                y_true = yy
                y_pred = cross_val_predict(est, XX, yy, cv=cv, method='predict')
                proba_pred = cross_val_predict(est, XX, yy, cv=cv, method='predict_proba')

                score = calc_score(scorer_name, y_true, y_pred, proba_pred,
                                   labels=config.LABELS, cost_mtx=config.COST_MTX, extra_args=scorer_args)
                return score

            score_pred = f_scoring(estimator, X_sel, y)
            score_penal = 1.0 - float(num_dims_sel) / float(num_dims_orig)
            score = (1.0 - penaliz) * score_pred + penaliz * score_penal

            loss = -1.0 * score  # PSO minimizes
            return loss

        # run PSO
        num_dims_orig = X.shape[1]
        f_optim_pso = lambda partic: f_loss_particle(partic,
                                                     method_wrap=self.method_wrap,
                                                     support_categ_feats=self.support_categ_feats,
                                                     num_cv_folds=self.num_cv_folds, penaliz=self.penaliz,
                                                     score_opts=self.score_opts)

        self.optimizer.initialize(num_dims_orig, f_optim_pso)
        _, support_mask = self.optimizer.optimize()

        self.fitted = True
        self.mask = support_mask
        return self


class FeatSelPSORegr(FeatSelPSO):
    def __init__(self, hyperparams, score_opts, random_seq):
        super(FeatSelPSORegr, self).__init__(hyperparams, score_opts, random_seq)

    def fit(self, X, y, **fit_params):
        # assemble all elements necessary to create the specific loss function
        random_st_bal, random_st_clf, random_st_cv = self.random_seq.spawn(1)[0].generate_state(3)
        balancer = RandomOverSampler(sampling_strategy='not majority', shrinkage=None,
                                     random_state=random_st_bal)
        regressor = None
        if self.method_wrap == 'linear':
            regressor = Ridge(**self.hyperparams_wrap, random_state=random_st_clf)
        elif self.method_wrap == 'svm':
            regressor = SVR(**self.hyperparams_wrap)
        elif self.method_wrap == 'hgb':
            regressor = HistGradientBoostingRegressor(**self.hyperparams_wrap, random_state=random_st_clf)
        if regressor is None:
            raise RuntimeError
        estimator = ImbalancedPipeline([('balancer', balancer), ('regressor', regressor)])

        # define loss
        def f_loss_particle(particle, method_wrap, support_categ_feats, num_cv_folds, penaliz, score_opts):
            part_mask = np.asarray(particle, dtype=bool)
            X_sel = X[:, part_mask]
            num_dims_orig = X.shape[1]
            num_dims_sel = X_sel.shape[1]

            if method_wrap == 'hgb' and support_categ_feats:
                feats_categ_aux = np.asarray(self.feat_categ, dtype=bool)
                feats_categ_aux = feats_categ_aux[part_mask]
                feats_categ_aux = feats_categ_aux.tolist()

                estimator.named_steps['regressor'].set_params(categorical_features=feats_categ_aux)

            cv = StratifiedKFold(n_splits=num_cv_folds, shuffle=True, random_state=random_st_cv)

            def f_scoring(est, XX, yy):
                scorer_name = score_opts['scorer_name']
                scorer_args = score_opts['scorer_args']

                y_true = yy
                y_pred = cross_val_predict(est, XX, yy, cv=cv, method='predict')
                proba_pred = None

                discret_y = DiscretizerY(num_classes=config.NUM_CLASSES)
                y_pred = discret_y.transform(y_pred)

                score = calc_score(scorer_name, y_true, y_pred, proba_pred,
                                   labels=config.LABELS, cost_mtx=config.COST_MTX, extra_args=scorer_args)
                return score

            score_pred = f_scoring(estimator, X_sel, y)
            score_penal = 1.0 - float(num_dims_sel) / float(num_dims_orig)
            score = (1.0 - penaliz) * score_pred + penaliz * score_penal

            loss = -1.0 * score  # PSO minimizes
            return loss

        # run PSO
        num_dims_orig = X.shape[1]
        f_optim_pso = lambda partic: f_loss_particle(partic,
                                                     method_wrap=self.method_wrap,
                                                     support_categ_feats=self.support_categ_feats,
                                                     num_cv_folds=self.num_cv_folds, penaliz=self.penaliz,
                                                     score_opts=self.score_opts)

        self.optimizer.initialize(num_dims_orig, f_optim_pso)
        _, support_mask = self.optimizer.optimize()

        self.fitted = True
        self.mask = support_mask
        return self


class _PSOptimizer:
    def __init__(self, num_iters, num_parts, options, vel_clamp):
        self.num_iters = num_iters
        self.num_parts = num_parts
        self.options = options
        self.vel_clamp = vel_clamp

        self.num_dims = None
        self.f_loss_pso = None

    def initialize(self, num_dims, f_loss_pso):
        self.num_dims = num_dims
        self.f_loss_pso = f_loss_pso

        return self

    def optimize(self):
        def f_loss_swarm(swarm, f_loss_particle):
            losses = list(map(f_loss_particle, swarm))
            losses = np.array(losses)
            return losses
        f_loss = lambda swarm: f_loss_swarm(swarm, self.f_loss_pso)

        verbose = (config.VERBOSE >= 5)
        optimizer = BinaryPSO(n_particles=self.num_parts, dimensions=self.num_dims,
                              options=self.options, velocity_clamp=self.vel_clamp)
        loss_opt, particle_opt = optimizer.optimize(f_loss, iters=self.num_iters, verbose=verbose)
        return loss_opt, particle_opt
