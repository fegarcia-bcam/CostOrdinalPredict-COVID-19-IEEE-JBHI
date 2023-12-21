import os
import json

import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from utils_cost import gen_cost_matrix_asymmetric


# files and paths
PATH_DATA = '../../data/'
FILE_DATA_IN = os.path.join(PATH_DATA, 'data_COVID-19.csv')
FILE_DATA_SUPPL = os.path.join(PATH_DATA, 'data_COVID-19_suppl.csv')
FILE_VAR_NAMES = os.path.join(PATH_DATA, 'var_names.json')
FILE_FEAT_NAMES = os.path.join(PATH_DATA, 'feat_names.json')

PATH_RESULTS_CLASSIFY = '../../results/classify'
FILE_RESULTS_CLASSIFY = 'results_{}.json'

PATH_RESULTS_CALIBRATE = '../../results/calibrate'
FILE_RESULTS_CALIBRATE = 'results_{}.json'

PATH_RESULTS_SHAP = '../../results/shap'
FILE_RESULTS_SHAP = 'results_{}.json'

PATH_RESULTS_BASELINE = '../../results/baseline'
FILE_RESULTS_BASELINE = 'results_{}.json'

DATETIME_FORMAT = '%Y-%m-%d_%H;%M;%S'
DELIMITER = ';'

# error message due to data availability
MSSG_ERROR_DATA = 'The dataset which supports the findings of this study could not be made publicly available due to restrictions from our Ethics Committees for Clinical Research (CEIc), ' \
                  'as it contains potentially identifying or sensitive patient information.\n' \
                  'For further requests, please contact CEIc at mailto:hgu.ceic@osakidetza.eus'

# data
# ordinal regression / ordinal classification problem
NUM_CLASSES = 3
LABELS = pd.Series(list(range(NUM_CLASSES)))

# three-class ordinal problem
COST_ALPHA = 3.0 / 2.0
COST_BETA = 4.0 / 3.0
COST_KAPPA = 3.0 / 2.0

COST_MTX = gen_cost_matrix_asymmetric(NUM_CLASSES, alpha=COST_ALPHA, beta=COST_BETA, kappa=COST_KAPPA)

with open(FILE_VAR_NAMES, 'r') as f_vars:
    VAR_NAMES = json.load(f_vars)

# stratification
VAR_GROUP = 'hospital'
VAR_CLASSIF = 'severity_ordinal'
VARS_STRATIF = [VAR_GROUP, VAR_CLASSIF]

VARS_EXTRA = ['severity_binary', 'death']

# decide whether hospital should be provided to the ML system or not
# EXCLUDE_HOSPITAL = False
EXCLUDE_HOSPITAL = True

# variable encoding
VARS_IN = [var for var in VAR_NAMES.values() if var not in VARS_STRATIF + VARS_EXTRA]

VARS_ORDINAL = {'pat_alcohol': 3, 'pat_tobacco': 3, 'pneumo_curb65': 5, 'sepsis_qsofa': 4,
                'symptoms_fever': 3, 'covid-treatm_cortic-iv': 3, 'covid-treatm_lmwh': 5}

VARS_CATEGORICAL = {'comorb_broncho': ['No', 'Asthma', 'DPLP', 'COPD', 'Others'],  # never missing
                    'emerg-treatm_cortico': ['No', 'Inhaled', 'Oral', 'Missing'],
                    'emerg-pulmo_infiltr-xr': ['No', 'Unilobar', 'Multilob unilat', 'Bilateral', 'Missing'],
                    'emerg-pulmo_infiltr-type': ['No', 'Alveolar', 'Consolidation', 'Interstitial', 'Missing'],
                    'covid-diagn_method': ['Fast serology', 'PCR sputum', 'PCR nasophar', 'Missing'],
                    'covid-treatm_antibiot': ['No', 'Beta-lactam', 'Macrolides', 'Macrol & Beta', 'Quinolones', 'Others']}  # never missing
if EXCLUDE_HOSPITAL:
    if VAR_GROUP in VARS_CATEGORICAL:
        VARS_CATEGORICAL.pop(VAR_GROUP)

VARS_CONTINUOUS = ['pat_age', 'pat_height', 'pat_weight', 'pat_bmi', 'comorb_charlson', 'pneumo_psi-sc',
                   'symptoms_days', 'emerg-status', 'emerg-pulmo_infiltr-lobs', 'blood-t', 'abgt', 'covid-diagn_days',
                   'demograph', 'pollut']

VARS_DISCRETE = []
for var in VARS_IN:
    is_discrete = True
    for var_heading in VARS_CONTINUOUS:
        if var.startswith(var_heading):
            is_discrete = False
            break
    VARS_DISCRETE.append(is_discrete)

l_transf_onehot = []
for variable in VARS_IN:
    if variable in VARS_CATEGORICAL.keys():
        categories = VARS_CATEGORICAL[variable]

        enc_onehot = OneHotEncoder(categories=[categories], drop='first', sparse=False, dtype='int')
        transf_onehot = (variable, enc_onehot, [variable])
        l_transf_onehot.append(transf_onehot)

    else:
        transf_pass = (variable, 'passthrough', [variable])
        l_transf_onehot.append(transf_pass)
ENCODER_ONEHOT = ColumnTransformer(l_transf_onehot)

# calibration
CALIBR_METHOD = 'sigmoid'
# CALIBR_METHOD = 'isotonic'

CALIBR_ENSEMBLE = True
# CALIBR_ENSEMBLE = False


# SHAP explanations
# K_SUMMARY_SHAP = 50
K_SUMMARY_SHAP = 100
# K_SUMMARY_SHAP = np.inf

# NUM_EVALS_SHAP = 256
NUM_EVALS_SHAP = 1024
# NUM_EVALS_SHAP = 'auto'


# choose parallelization
# execution
# PARALLEL_CLASSIF_REPEAT = True
PARALLEL_CLASSIF_REPEAT = False
NUM_NODES_CLASSIF_REPEAT = 20
# NUM_NODES_CLASSIF_REPEAT = 10

# PARALLEL_CLASSIF_OUT_CV = True
PARALLEL_CLASSIF_OUT_CV = False
# NUM_NODES_CLASSIF_OUT_CV = 10
NUM_NODES_CLASSIF_OUT_CV = 5

# PARALLEL_CLASSIF_HYPERPARAM_OPT = True
PARALLEL_CLASSIF_HYPERPARAM_OPT = False
# change number of computation nodes, in relation to the size of the hyperparameter search space
# NUM_NODES_CLASSIF_HYPERPARAM_OPT = 12
NUM_NODES_CLASSIF_HYPERPARAM_OPT = 18

# PARALLEL_CLASSIF_IN_CV = True
PARALLEL_CLASSIF_IN_CV = False
# NUM_NODES_CLASSIF_IN_CV = 10
NUM_NODES_CLASSIF_IN_CV = 5

parallel_flags_classif = pd.Series([PARALLEL_CLASSIF_REPEAT, PARALLEL_CLASSIF_OUT_CV, PARALLEL_CLASSIF_HYPERPARAM_OPT, PARALLEL_CLASSIF_IN_CV], dtype=bool)
if parallel_flags_classif.sum() > 1:
    raise RuntimeError  # nested parallelization is not supported by Pathos
parallel_nodes_classif = pd.Series([NUM_NODES_CLASSIF_REPEAT, NUM_NODES_CLASSIF_OUT_CV, NUM_NODES_CLASSIF_HYPERPARAM_OPT, NUM_NODES_CLASSIF_IN_CV], dtype=int)
requested_nodes_classif = parallel_nodes_classif.multiply(parallel_flags_classif).max()
requested_nodes_classif = max(1, requested_nodes_classif)


# PARALLEL_CALIBR_REPEAT = True
PARALLEL_CALIBR_REPEAT = False
NUM_NODES_CALIBR_REPEAT = 20
# NUM_NODES_CALIBR_REPEAT = 10

# PARALLEL_CALIBR_CV = True
PARALLEL_CALIBR_CV = False
NUM_NODES_CALIBR_CV = 10
# NUM_NODES_CALIBR_CV = 5

parallel_flags_calibr = pd.Series([PARALLEL_CALIBR_REPEAT, PARALLEL_CALIBR_CV], dtype=bool)
if parallel_flags_calibr.sum() > 1:
    raise RuntimeError  # nested parallelization is not supported by Pathos
parallel_nodes_calibr = pd.Series([NUM_NODES_CALIBR_REPEAT, NUM_NODES_CALIBR_CV], dtype=int)
requested_nodes_calibr = parallel_nodes_calibr.multiply(parallel_flags_calibr).max()


# PARALLEL_SHAP = True
PARALLEL_SHAP = False


NUM_JOBS_RELIEF = None
# NUM_JOBS_RELIEF = -1
# NUM_JOBS_RELIEF = requested_nodes_classif

NUM_JOBS_RFECV = None
# NUM_JOBS_RFECV = -1
# NUM_JOBS_RFECV = requested_nodes_classif

NUM_JOBS_SEQ_FEAT_SEL = None
# NUM_JOBS_SEQ_FEAT_SEL = -1
# NUM_JOBS_SEQ_FEAT_SEL = requested_nodes_classif

NUM_JOBS_IMBALANCE = None
# NUM_JOBS_IMBALANCE = -1
# NUM_JOBS_IMBALANCE = requested_nodes_classif

NUM_JOBS_RANDOM_FOREST = None
# NUM_JOBS_RANDOM_FOREST = -1
# NUM_JOBS_RANDOM_FOREST = requested_nodes_classif

NUM_JOBS_LOGIST_REG = None
# NUM_JOBS_LOGIST_REG = -1
# NUM_JOBS_LOGIST_REG = requested_nodes_classif

NUM_JOBS_FEAT_PERMUTS = None
# NUM_JOBS_FEAT_PERMUTS = -1
# NUM_JOBS_FEAT_PERMUTS = requested_nodes_classif

# display
CLASS_NAMES = ['Low', 'Med', 'High']
CLASS_COLOURS = ['green', 'orange', 'red']

# TAGS_TRUE = ['True #{}'.format(i) for i in range(NUM_CLASSES)]
# TAGS_PRED = ['Pred #{}'.format(i) for i in range(NUM_CLASSES)]
TAGS_TRUE = ['True {}'.format(CLASS_NAMES[i]) for i in range(NUM_CLASSES)]
TAGS_PRED = ['Pred {}'.format(CLASS_NAMES[i]) for i in range(NUM_CLASSES)]

# verbosity
# VERBOSE = 0  # nothing
# VERBOSE = 1  # best hyperparameters, outer cv folds
# VERBOSE = 2  # inner cv aggregate
# VERBOSE = 3  # inner cv folds
# VERBOSE = 4  # feature selection results
VERBOSE = 5  # feature selection intermediate

# randomization
SHUFFLE = True
# SHUFFLE = False

SEED_CLASSIFY = None
# SEED_CLASSIFY = 0
