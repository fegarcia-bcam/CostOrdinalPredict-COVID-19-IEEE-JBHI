import warnings
import copy
import random

from abc import abstractmethod

import numpy as np
from tqdm import tqdm

from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC, SVR

from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from sklearn.model_selection import StratifiedKFold, cross_val_predict

from imblearn.over_sampling import RandomOverSampler

from deap import base, creator, tools

from Discretizer import DiscretizerY
from ImbalancedPipeline import ImbalancedPipeline

from calc_scoring import calc_score

from FeatureSelection import WRAP_METHODS

import config


class FeatSelGA(SelectorMixin, BaseEstimator):
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
        hyperparams_remove = ['num_iters', 'num_indivs', 'options']
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
        num_indivs = self.hyperparams['num_indivs']
        options = self.hyperparams['options']

        optimizer = _GAOptimizer(num_iters=num_iters, num_indivs=num_indivs, options=options)
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


class FeatSelGAClass(FeatSelGA):
    def __init__(self, hyperparams, score_opts, random_seq):
        super(FeatSelGAClass, self).__init__(hyperparams, score_opts, random_seq)

    def fit(self, X, y, **fit_params):
        # assemble all elements necessary to create the specific fitness function
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

        # define fitness
        def f_fitness_indiv(individual, method_wrap, support_categ_feats, num_cv_folds, penaliz, score_opts):
            indiv_mask = np.asarray(individual, dtype=bool)
            X_sel = X[:, indiv_mask]
            num_dims_orig = X.shape[1]
            num_dims_sel = X_sel.shape[1]

            if method_wrap == 'hgb' and support_categ_feats:
                feats_categ_aux = np.asarray(self.feat_categ, dtype=bool)
                feats_categ_aux = feats_categ_aux[indiv_mask]
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

            fitness = score
            return fitness,  # attention: needs to return a tuple!!!

        # run GA
        num_dims_orig = X.shape[1]
        f_optim_ga = lambda indiv: f_fitness_indiv(indiv,
                                                   method_wrap=self.method_wrap,
                                                   support_categ_feats=self.support_categ_feats,
                                                   num_cv_folds=self.num_cv_folds, penaliz=self.penaliz,
                                                   score_opts=self.score_opts)

        self.optimizer.initialize(num_dims_orig, f_optim_ga)
        _, support_mask = self.optimizer.optimize()

        self.fitted = True
        self.mask = support_mask
        return self


class FeatSelGARegr(FeatSelGA):
    def __init__(self, hyperparams, score_opts, random_seq):
        super(FeatSelGARegr, self).__init__(hyperparams, score_opts, random_seq)

    def fit(self, X, y, **fit_params):
        # assemble all elements necessary to create the specific fitness function
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

        # define fitness
        def f_fitness_indiv(individual, method_wrap, support_categ_feats, num_cv_folds, penaliz, score_opts):
            indiv_mask = np.asarray(individual, dtype=bool)
            X_sel = X[:, indiv_mask]
            num_dims_orig = X.shape[1]
            num_dims_sel = X_sel.shape[1]

            if method_wrap == 'hgb' and support_categ_feats:
                feats_categ_aux = np.asarray(self.feat_categ, dtype=bool)
                feats_categ_aux = feats_categ_aux[indiv_mask]
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

            fitness = score
            return fitness,  # attention: needs to return a tuple!!!

        # run GA
        num_dims_orig = X.shape[1]
        f_optim_ga = lambda indiv: f_fitness_indiv(indiv,
                                                   method_wrap=self.method_wrap,
                                                   support_categ_feats=self.support_categ_feats,
                                                   num_cv_folds=self.num_cv_folds, penaliz=self.penaliz,
                                                   score_opts=self.score_opts)

        self.optimizer.initialize(num_dims_orig, f_optim_ga)
        _, support_mask = self.optimizer.optimize()

        self.fitted = True
        self.mask = support_mask
        return self


class _GAOptimizer:
    def __init__(self, num_iters, num_indivs, options):
        self.num_iters = num_iters
        self.num_indivs = num_indivs
        self.options = options

        self.feat_names = None
        self.feat_categ = None

        self.num_dims = None
        self.f_fitness_ga = None

        self.toolbox = None
        self.population = None
        self.fitness_all = None

        self.prob_flip = self.options['prob_flip']
        self.prob_cross = self.options['prob_cross']
        self.prob_mutat = self.options['prob_mutat']
        self.select_type = self.options['select_type']
        self.select_param = self.options['select_param']

    def initialize(self, num_dims, f_fitness_ga):
        self.num_dims = num_dims
        self.f_fitness_ga = f_fitness_ga

        # define framework
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            creator.create('FeatureSelect', base.Fitness, weights=(1.0,))
            creator.create('Individual', list, fitness=creator.FeatureSelect)

        # create toolbox
        toolbox = base.Toolbox()

        # define each gene as a discrete binary random feature
        toolbox.register('attr_bool', random.randint, 0, 1)

        # define the individual to be a set of genes
        toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.attr_bool, self.num_dims)

        # define the population to be a list of individuals
        toolbox.register('population', tools.initRepeat, list, toolbox.individual)

        # define the fitness function
        toolbox.register('evaluate', self.f_fitness_ga)

        # define the crossover operator
        toolbox.register('mate', tools.cxTwoPoint)

        # define a mutation operator with a probability to flip each gene
        toolbox.register('mutate', tools.mutFlipBit, indpb=self.prob_flip)

        # define the operator for selecting individuals for breeding the next generation
        if self.select_type == 'roulette':
            # the probability of choosing an individual is proportional to its fitness
            toolbox.register('select', tools.selRoulette)
        elif self.select_type == 'tournament':
            # each individual of the current generation is replaced by the 'fittest'
            # of tournsize individuals, drawn randomly from the current generation
            toolbox.register('select', tools.selTournament, tournsize=self.select_param)
        elif self.select_type == 'nsga2':
            # NSGA-II selection
            toolbox.register('select', tools.selNSGA2, nd=self.select_param)
        else:
            raise NotImplementedError

        self.toolbox = toolbox
        return self

    def optimize(self):
        verbose = (config.VERBOSE >= 5)

        # create an initial population and evaluate it
        self.population = self.toolbox.population(n=self.num_indivs)
        self.fitness_all = []
        if verbose:
            print('Genetic initialization...', flush=True)
        fitness_pop = list(map(self.toolbox.evaluate, self.population))
        for indiv, fit in zip(self.population, fitness_pop):
            indiv.fitness.values = fit

        fitness_gen = [indiv.fitness.values[0] for indiv in self.population]
        self.fitness_all.append(fitness_gen)

        # run evolution generations
        if verbose:
            for _ in tqdm(range(self.num_iters), desc='Genetic evolution', total=self.num_iters):
                self.run_generation()
        else:
            for _ in range(self.num_iters):
                self.run_generation()

        # select the best individual
        indiv_opt = tools.selBest(self.population, 1)[0]
        fitness_opt = indiv_opt.fitness.values
        indiv_opt = np.asarray(list(indiv_opt))

        return fitness_opt, indiv_opt

    def run_generation(self):
        # select the next generation individuals
        offspring = self.toolbox.select(self.population, len(self.population))
        # clone the selected individuals
        offspring = list(map(self.toolbox.clone, offspring))

        # apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            # cross two individuals with crossover probability
            if random.random() < self.prob_cross:
                self.toolbox.mate(child1, child2)
                # fitness values of the children must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            # mutate an individual with mutation probability
            if random.random() < self.prob_mutat:
                self.toolbox.mutate(mutant)
                # fitness values of the mutant must be recalculated later
                del mutant.fitness.values

        # reevaluate the individuals with an invalid fitness, as it changed from previous generation
        indiv_weak = [indiv for indiv in offspring if not indiv.fitness.valid]
        fitness_weak = map(self.toolbox.evaluate, indiv_weak)
        for indiv, fit in zip(indiv_weak, fitness_weak):
            indiv.fitness.values = fit

        # the population is entirely replaced by the offspring
        self.population[:] = offspring

        # gather all the fitnesses in one list
        fitness_gen = [indiv.fitness.values[0] for indiv in self.population]
        self.fitness_all.append(fitness_gen)

        return self
