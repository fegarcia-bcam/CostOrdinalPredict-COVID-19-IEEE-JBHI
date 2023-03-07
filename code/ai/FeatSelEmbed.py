from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin

from sklearn.feature_selection import SelectFromModel

from sklearn.linear_model import LogisticRegression, Lasso

from imblearn.over_sampling import RandomOverSampler

from ImbalancedPipeline import ImbalancedPipeline


class FeatSelL1Classif(SelectorMixin, BaseEstimator):
    def __init__(self, hyperparams, random_seq):
        self.hyperparams = hyperparams
        self.random_seq = random_seq

        self.feat_names = None
        self.feat_categ = None

        self.fitted = False
        self.mask = None

        random_st_bal, random_st_clf = self.random_seq.spawn(1)[0].generate_state(2)

        # build
        balancer = RandomOverSampler(sampling_strategy='not majority', shrinkage=None,
                                     random_state=random_st_bal)

        C = self.hyperparams['C']
        classifier = LogisticRegression(C=C, penalty='l1', multi_class='multinomial',
                                        solver='saga', max_iter=int(1e4), random_state=random_st_clf)

        estimator = ImbalancedPipeline([('balancer', balancer), ('classifier', classifier)])

        importance_getter = 'named_steps.classifier.coef_'

        core = SelectFromModel(estimator=estimator, importance_getter=importance_getter)
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


class FeatSelL1Regress(SelectorMixin, BaseEstimator):
    def __init__(self, hyperparams, random_seq):
        self.hyperparams = hyperparams
        self.random_seq = random_seq

        self.feat_names = None
        self.feat_categ = None

        self.fitted = False
        self.mask = None

        random_st_bal, random_st_reg = self.random_seq.spawn(1)[0].generate_state(2)

        # build
        balancer = RandomOverSampler(sampling_strategy='not majority', shrinkage=None,
                                     random_state=random_st_bal)

        alpha = self.hyperparams['alpha']
        regressor = Lasso(alpha=alpha,
                          max_iter=int(1e4), random_state=random_st_reg)

        estimator = ImbalancedPipeline([('balancer', balancer), ('regressor', regressor)])

        importance_getter = 'named_steps.regressor.coef_'

        core = SelectFromModel(estimator=estimator, importance_getter=importance_getter)
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
