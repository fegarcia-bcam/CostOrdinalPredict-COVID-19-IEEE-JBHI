import numpy as np
import pandas as pd

from numpy.random import SeedSequence

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.base import clone

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from utils_proba import PROBA_DECISION
from utils_proba import bound_proba, classify_proba
from calc_scoring import calc_score

from FeatureSelection import FeatureSelector, FEATURE_SELECTORS

from OrdinalScenario import OrdinalScenario
from OrdinalDeReComp import ORDINAL_DECOMPOSERS
from OrdinalDeReCompRegress import OrdinalDeReCompRegress
from OrdinalDeReCompOneVsOne import OrdinalDeReCompOneVsOne
from OrdinalDeReCompOrderPart import OrdinalDeReCompOrderPart

import config

PIPELINE_STAGES = ['encoder', 'scaler', 'imputer', 'feat_sel', 'decompos', 'balancer', 'estimator']

ENCODERS_AVAIL = ['onehot']
SCALERS_AVAIL = ['none', 'standard', 'robust']
IMPUTERS_AVAIL = ['none', 'simple', 'knn', 'iterative']
FEAT_SELS_AVAIL = ['none'] + FEATURE_SELECTORS
DECOMPOS_AVAIL = ORDINAL_DECOMPOSERS
ESTIMATORS_AVAIL = ['linear', 'svm', 'mlp', 'rf', 'ab', 'gb', 'hgb']


class OrdinalClassifPipeline(OrdinalScenario):

    random_seq = None

    decisor = None
    scorer = None

    options = None
    hyperparams = None

    encoder = None
    scaler = None
    imputer = None
    feature_selector = None
    ordinal_derecompos = None
    balancers = None
    estimators = None

    def __init__(self, decisor, scorer, options, hyperparams, random_seq,
                 num_classes=config.NUM_CLASSES, labels=config.LABELS, cost_mtx=config.COST_MTX):
        self.set_random_seq(random_seq)
        self.configure(num_classes=num_classes, labels=labels, cost_mtx=cost_mtx,
                       decisor=decisor, scorer=scorer, options=options, hyperparams=hyperparams)

    def get_decisor(self):
        return self.decisor

    def set_decisor(self, decisor):
        if not isinstance(decisor, str):
            raise TypeError
        if decisor not in PROBA_DECISION:
            raise ValueError
        self.decisor = decisor
        return self

    def get_scorer(self):
        return self.scorer

    def set_scorer(self, scorer):
        if not isinstance(scorer, dict):
            raise TypeError
        if set(list(scorer.keys())) != {'scorer_name', 'scorer_args'}:
            raise ValueError
        self.scorer = scorer
        return self

    def get_options(self):
        return self.options

    def set_options(self, options):
        if not isinstance(options, dict):
            raise TypeError
        if set(list(options.keys())) != set(PIPELINE_STAGES):
            raise ValueError
        self.options = options
        return self

    def get_hyperparams(self):
        return self.hyperparams

    def set_hyperparams(self, hyperparams):
        if not isinstance(hyperparams, dict):
            raise TypeError
        if set(list(hyperparams.keys())) != set(PIPELINE_STAGES):
            raise ValueError
        self.hyperparams = hyperparams
        return self

    def get_random_seq(self):
        return self.random_seq

    def set_random_seq(self, random_seq):
        if not isinstance(random_seq, SeedSequence):
            raise TypeError
        self.random_seq = random_seq
        return self

    def get_params(self, deep=True):
        params = {'num_classes': self.get_num_classes(),
                  'labels': self.get_labels(),
                  'cost_mtx': self.get_cost_matrix(),
                  'decisor': self.get_decisor(),
                  'scorer': self.get_scorer(),
                  'options': self.get_options(),
                  'hyperparams': self.get_hyperparams(),
                  'random_seq': self.get_random_seq()}

        return params

    def set_params(self, hyperparams):
        self.set_hyperparams(hyperparams)
        self.configure(self.get_num_classes(), self.get_labels(),
                       self.get_cost_matrix(), self.get_scorer(),
                       self.get_options(), hyperparams)
        return self

    def get_stages(self):
        return self.encoder, self.scaler, self.imputer, self.feature_selector, \
               self.ordinal_derecompos, self.balancers, self.estimators

    def configure(self, num_classes, labels, cost_mtx, decisor, scorer, options, hyperparams):
        self.set_num_classes(num_classes)
        self.set_labels(labels)
        self.set_cost_matrix(cost_mtx)

        self.set_decisor(decisor)
        self.set_scorer(scorer)

        self.set_options(options)
        self.set_hyperparams(hyperparams)

        self.initialize()
        return self

    def initialize(self):
        self.set_encoder()
        self.set_scaler()
        self.set_imputer()
        self.set_feature_selector()
        self.set_ordinal_derecompos()
        self.set_balancers(balancers=None)  # need to be instantiated at fitting time
        self.set_estimators()
        return self

    def set_encoder(self):
        options = self.get_options()
        hyperparams = self.get_hyperparams()
        encoder_str = options['encoder']
        encoder_hyp = hyperparams['encoder']  # meaningless

        encoder = None
        if encoder_str not in ENCODERS_AVAIL:
            raise NotImplementedError
        elif encoder_str == 'onehot':
            encoder = config.ENCODER_ONEHOT
        self.encoder = encoder
        return self

    def set_scaler(self):
        options = self.get_options()
        hyperparams = self.get_hyperparams()
        scaler_str = options['scaler']
        scaler_hyp = hyperparams['scaler']

        scaler = None
        if scaler_str not in SCALERS_AVAIL:
            raise NotImplementedError
        elif scaler_str == 'none':
            scaler = None
        elif scaler_str == 'standard':
            scaler = StandardScaler(**scaler_hyp)
        elif scaler_str == 'robust':
            scaler = RobustScaler(**scaler_hyp)
        self.scaler = scaler
        return self

    def set_imputer(self):
        options = self.get_options()
        hyperparams = self.get_hyperparams()
        imputer_str = options['imputer']
        imputer_hyp = hyperparams['imputer']

        imputer = None
        if imputer_str not in IMPUTERS_AVAIL:
            raise NotImplementedError
        elif imputer_str == 'none':
            pass
        elif imputer_str == 'simple':
            imputer = SimpleImputer(**imputer_hyp)
        elif imputer_str == 'knn':
            imputer = KNNImputer(**imputer_hyp)
        elif imputer_str == 'iterative':
            # generate random state: first create child sequence by spawning, then draw a sample
            random_seq = self.get_random_seq()
            random_state = random_seq.spawn(1)[0].generate_state(1)[0]

            imputer = IterativeImputer(random_state=random_state, **imputer_hyp)
        self.imputer = imputer
        return self

    def set_feature_selector(self):
        options = self.get_options()
        hyperparams = self.get_hyperparams()
        feat_sel_method = options['feat_sel']['method']
        feat_sel_scoring = options['feat_sel']['scoring']
        feat_sel_hyp = hyperparams['feat_sel']

        feature_selector = None
        if feat_sel_method not in FEAT_SELS_AVAIL:
            raise NotImplementedError
        elif feat_sel_method == 'none':
            pass
        else:
            # generate child random sequence
            random_seq_mother = self.get_random_seq()
            random_seq_child = random_seq_mother.spawn(1)[0]

            feature_selector = FeatureSelector(method=feat_sel_method, hyperparams=feat_sel_hyp,
                                               score_opts=feat_sel_scoring, random_seq=random_seq_child)
        self.feature_selector = feature_selector
        return self

    def set_ordinal_derecompos(self):
        options = self.get_options()
        decompos_type = options['decompos']['type']
        decompos_policy = options['decompos']['policy']

        ordinal_derecompos = None
        if decompos_type not in DECOMPOS_AVAIL:
            raise ValueError
        elif decompos_type == 'regress':
            ordinal_derecompos = OrdinalDeReCompRegress(policy=decompos_policy,
                num_classes=self.get_num_classes(), labels=self.get_labels(), cost_mtx=self.get_cost_matrix())
        elif decompos_type == 'one-vs-one':
            ordinal_derecompos = OrdinalDeReCompOneVsOne(policy=decompos_policy,
                num_classes=self.get_num_classes(), labels=self.get_labels(), cost_mtx=self.get_cost_matrix())
        elif decompos_type == 'ordered-part':
            ordinal_derecompos = OrdinalDeReCompOrderPart(policy=decompos_policy,
                num_classes=self.get_num_classes(), labels=self.get_labels(), cost_mtx=self.get_cost_matrix())

        self.ordinal_derecompos = ordinal_derecompos
        return self

    def set_balancers(self, balancers):
        if balancers is None:
            _, _, _, _, ordinal_derecompos, _, _ = self.get_stages()
            num_decomp = ordinal_derecompos.get_num_decomp()
            balancers = [None] * num_decomp  # need to be instantiated at fitting time

        self.balancers = balancers
        return self

    def set_estimators(self):
        options = self.get_options()
        hyperparams = self.get_hyperparams()
        decompos_type = options['decompos']['type']
        estimator_str = options['estimator']
        estimator_hyp = hyperparams['estimator']

        _, _, _, _, ordinal_derecompos, _, _ = self.get_stages()
        num_decomp = ordinal_derecompos.get_num_decomp()

        # generate random states: first create child sequence by spawning, then draw num_decomp samples
        random_seq = self.get_random_seq()
        random_states_estim = random_seq.spawn(1)[0].generate_state(num_decomp)

        if decompos_type not in DECOMPOS_AVAIL:
            raise NotImplementedError
        if estimator_str not in ESTIMATORS_AVAIL:
            raise NotImplementedError

        estimators = []
        if decompos_type == 'regress':  # regression decomposes in a single problem
            estimators = []
            for random_state in random_states_estim:
                estim = None
                if estimator_str == 'linear':
                    estim = Ridge(random_state=random_state)  # no support for multiple jobs

                elif estimator_str == 'svm':
                    estim = SVR()  # no support for multiple jobs, no randomness

                elif estimator_str == 'mlp':
                    estim = MLPRegressor(random_state=random_state)  # no support for multiple jobs

                elif estimator_str == 'rf':
                    estim = RandomForestRegressor(random_state=random_state, n_jobs=config.NUM_JOBS_RANDOM_FOREST)

                elif estimator_str == 'ab':
                    # generate base estimator
                    base_estim_str = estimator_hyp['base_estim']
                    if base_estim_str != 'decision_tree':
                        raise NotImplementedError
                    max_depth = estimator_hyp['max_depth']
                    random_state_base = random_seq.spawn(1)[0].generate_state(1)[0]  # randomness
                    base_estim = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state_base)

                    estim = AdaBoostRegressor(base_estim, random_state=random_state)  # no support for multiple jobs

                elif estimator_str == 'gb':
                    estim = GradientBoostingRegressor(random_state=random_state)  # no support for multiple jobs

                elif estimator_str == 'hgb':
                    estim = HistGradientBoostingRegressor(random_state=random_state)  # no support for multiple jobs

                estimators.append(estim)

        elif decompos_type in DECOMPOS_AVAIL:  # either 'one-vs-one' or 'ordered-part'
            estimators = []
            for random_state in random_states_estim:
                estim = None
                if estimator_str == 'linear':
                    estim = LogisticRegression(random_state=random_state, n_jobs=config.NUM_JOBS_LOGIST_REG)

                elif estimator_str == 'svm':
                    estim = SVC(random_state=random_state)  # no support for multiple jobs

                elif estimator_str == 'mlp':
                    estim = MLPClassifier(random_state=random_state)  # no support for multiple jobs

                elif estimator_str == 'rf':
                    estim = RandomForestClassifier(random_state=random_state, n_jobs=config.NUM_JOBS_RANDOM_FOREST)

                elif estimator_str == 'ab':
                    # generate base estimator
                    base_estim_str = estimator_hyp['base_estim']
                    if base_estim_str != 'decision_tree':
                        raise NotImplementedError
                    max_depth = estimator_hyp['max_depth']
                    random_state_base = random_seq.spawn(1)[0].generate_state(1)[0]  # randomness
                    base_estim = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state_base)

                    estim = AdaBoostClassifier(base_estim, random_state=random_state)  # no support for multiple jobs

                elif estimator_str == 'gb':
                    estim = GradientBoostingClassifier(random_state=random_state)  # no support for multiple jobs

                elif estimator_str == 'hgb':
                    estim = HistGradientBoostingClassifier(random_state=random_state)  # no support for multiple jobs

                estimators.append(estim)

        if estimator_str == 'ab':
            estimator_hyp_ab = estimator_hyp.copy()
            estimator_hyp_ab.pop('base_estim')
            estimator_hyp_ab.pop('max_depth')
            for est in estimators:
                est.set_params(**estimator_hyp_ab)
        elif estimator_str == 'hgb':
            estimator_hyp_hgb = estimator_hyp.copy()
            support_categ_feats = estimator_hyp_hgb.pop('support_categ_feats', False)
            for est in estimators:
                est.set_params(**estimator_hyp_hgb)
                if support_categ_feats:
                    est.set_params(categorical_features=config.VARS_DISCRETE)
        else:
            for est in estimators:
                est.set_params(**estimator_hyp)

        self.estimators = estimators
        return self

    def fit_pipe(self, X_train, y_train):
        self.classes_ = self.get_labels().tolist()

        options = self.get_options()
        hyperparams = self.get_hyperparams()
        random_seq = self.get_random_seq()
        encoder, scaler, imputer, feature_selector, ordinal_derecompos, _, _ = self.get_stages()

        # feature names and whether categorical, before encoding
        feat_names = X_train.columns.tolist()
        feat_categ = config.VARS_DISCRETE

        # encoder
        X_train = encoder.fit_transform(X_train)

        # feature names and whether categorical, after encoding
        try:
            feat_names_enc = encoder.get_feature_names_out().tolist()
            feat_names_enc = [f_name.split('__')[-1].lower() for f_name in feat_names_enc]

            feat_categ_enc = []
            for f_name_enc in feat_names_enc:
                for f_name_orig, is_categ_orig in zip(feat_names, feat_categ):
                    if f_name_enc.startswith(f_name_orig):
                        is_categ_enc = is_categ_orig
                        feat_categ_enc.append(is_categ_enc)
                        break

            feat_names = feat_names_enc
            feat_categ = feat_categ_enc

        except AttributeError:
            pass

        # scaler
        if scaler is not None:
            l_scale = []
            scaler_base = clone(scaler)

            for feat_name in feat_names:
                to_scale = False
                # scale all continuous features
                for feat_contin in config.VARS_CONTINUOUS:
                    if feat_name.startswith(feat_contin):
                        to_scale = True
                        break
                # scale ordinal features, but not those induced as ordinal by the encoding
                if feat_name in config.VARS_ORDINAL.keys():
                    to_scale = True

                if to_scale:
                    transf_scale = (feat_name, clone(scaler_base), [feat_name])
                    l_scale.append(transf_scale)
                else:
                    transf_pass = (feat_name, 'passthrough', [feat_name])
                    l_scale.append(transf_pass)

            scaler = ColumnTransformer(l_scale)

            X_train = pd.DataFrame(data=X_train, columns=feat_names)
            X_train = scaler.fit_transform(X_train)

            self.scaler = scaler

        # imputer
        if imputer is not None:
            X_train = imputer.fit_transform(X_train)

            # feature names and whether categorical after imputation
            # include names for those extra features newly created by the missing indicator
            feat_names_imp = feat_names.copy()
            feat_categ_imp = feat_categ.copy()
            imputer_indic = imputer.indicator_
            if imputer_indic is not None:
                for idx_feat in imputer_indic.features_:
                    feat_name = feat_names[idx_feat] + config.MISSING_TAG
                    feat_names_imp.append(feat_name)
                    feat_categ_imp.append(True)

                feat_names = feat_names_imp
                feat_categ = feat_categ_imp

        # feature selector
        if feature_selector is not None:
            feature_selector.set_feature_config(feat_names, feat_categ)
            X_train = feature_selector.fit_transform(X_train, y_train)
            feat_mask = feature_selector.get_support()
            if config.VERBOSE >= 4:
                num_sel_feats = sum(feat_mask)
                print('Num. selected features: {}'.format(num_sel_feats))

            # feature names and whether categorical after selection
            feat_names_sel = []
            feat_categ_sel = []
            for f_sel, f_name, f_categ in zip(feat_mask, feat_names, feat_categ):
                if f_sel:
                    feat_names_sel.append(f_name)
                    feat_categ_sel.append(f_categ)

            feat_names = feat_names_sel
            feat_categ = feat_categ_sel

        # ordinal decomposition & sample balancing
        l_X_train_bin, l_y_train_bin, balancers = ordinal_derecompos.decompose(X_train, y_train,
                                                                               options, hyperparams, random_seq)
        # store generated sample balancers
        self.set_balancers(balancers)

        # fit estimator(s)
        _, _, _, _, _, _, estimators = self.get_stages()
        for est, X_train_bin, y_train_bin in zip(estimators, l_X_train_bin, l_y_train_bin):
            # update categorical features if supported by HistGradientBoosting estimators
            if type(est) in [HistGradientBoostingClassifier, HistGradientBoostingRegressor]:
                if est.get_params()['categorical_features'] is not None:
                    est.set_params(categorical_features=feat_categ)

            # train
            est.fit(X_train_bin, y_train_bin)  # training procedure itself

        return X_train

    def fit(self, X, y):
        self.fit_pipe(X, y)
        return self

    def fit_predict(self, X, y):
        self.fit_pipe(X, y)
        y_pred = self.predict(X)
        return y_pred

    def fit_transform(self, X, y):
        X = self.fit_pipe(X, y)
        return X

    def predict_pipe(self, X_test):
        encoder, scaler, imputer, feature_selector, ordinal_derecompos, _, estimators = self.get_stages()

        # feature names, before encoding
        feat_names = X_test.columns.tolist()

        # encoder
        X_test = encoder.transform(X_test)

        # feature names, after encoding
        try:
            feat_names_enc = encoder.get_feature_names_out().tolist()
            feat_names_enc = [f_name.split('__')[-1].lower() for f_name in feat_names_enc]

            feat_names = feat_names_enc

        except AttributeError:
            pass

        # scaler
        if scaler is not None:
            X_test = pd.DataFrame(data=X_test, columns=feat_names)
            X_test = scaler.transform(X_test)

        # imputer
        if imputer is not None:
            X_test = imputer.transform(X_test)

        # feature selector
        if feature_selector is not None:
            X_test = feature_selector.transform(X_test)

        # ordinal recomposition, no sample balancing
        if isinstance(ordinal_derecompos, OrdinalDeReCompRegress):
            est = estimators[0]
            values_test = est.predict(X_test)
            proba_test_pred = ordinal_derecompos.recompose(values_test)

        elif isinstance(ordinal_derecompos, (OrdinalDeReCompOneVsOne, OrdinalDeReCompOrderPart)):
            l_proba_bin = []
            for est in estimators:
                proba_bin = est.predict_proba(X_test)
                proba_bin = bound_proba(proba_bin)
                l_proba_bin.append(proba_bin)
            proba_test_pred = ordinal_derecompos.recompose(l_proba_bin)

        else:
            raise ValueError

        labels = self.get_labels()
        num_classes = self.get_num_classes()
        cost_mtx = self.get_cost_matrix()
        decisor = self.get_decisor()
        y_test_pred, proba_test_pred = classify_proba(proba_test_pred, labels=labels, cost_mtx=cost_mtx,
                                                      decision=decisor, num_classes=num_classes)

        return y_test_pred, proba_test_pred

    def predict(self, X):
        y_pred, _ = self.predict_pipe(X)
        return y_pred

    def predict_proba(self, X):
        _, proba_pred = self.predict_pipe(X)
        return proba_pred

    def predict_log_proba(self, X):
        _, proba_pred = self.predict_pipe(X)
        log_proba_pred = np.log(proba_pred)
        return log_proba_pred

    def score(self, X, y):
        labels = self.get_labels()
        cost_mtx = self.get_cost_matrix()

        scorer = self.get_scorer()
        scorer_name = scorer['scorer_name']
        scorer_args = scorer['scorer_args']

        # call prediction
        y_pred, proba_pred = self.predict_pipe(X)

        # calc score
        score = calc_score(scorer_name, y, y_pred, proba_pred,
                           labels=labels, cost_mtx=cost_mtx, extra_args=scorer_args)
        return score
