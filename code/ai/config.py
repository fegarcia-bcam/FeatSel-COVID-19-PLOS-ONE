import os
import json

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# files and paths
PATH_DATA = '../../data/'
FILE_DATA_IN = os.path.join(PATH_DATA, 'data_COVID-19.csv')
FILE_VAR_NAMES = os.path.join(PATH_DATA, 'var_names.json')
FILE_FEAT_NAMES = os.path.join(PATH_DATA, 'feat_names.json')

PATH_RESULTS = '../../results/feat_sel'
PATH_RESULTS_EMB = os.path.join(PATH_RESULTS, 'embedded')
PATH_RESULTS_FIL = os.path.join(PATH_RESULTS, 'filters')
PATH_RESULTS_WRA = os.path.join(PATH_RESULTS, 'wrappers')
PATH_RESULTS_WRA_SFS = os.path.join(PATH_RESULTS_WRA, 'sfs')
PATH_RESULTS_WRA_RFECV = os.path.join(PATH_RESULTS_WRA, 'rfecv')
PATH_RESULTS_WRA_RFE = os.path.join(PATH_RESULTS_WRA, 'rfe')
PATH_RESULTS_WRA_GA = os.path.join(PATH_RESULTS_WRA, 'ga')
PATH_RESULTS_WRA_BPSO = os.path.join(PATH_RESULTS_WRA, 'bpso')

PATH_RESULTS_STABLE = os.path.join(PATH_RESULTS, 'stable')

PATH_EXTRA = os.path.join(PATH_RESULTS, 'extra')
PATH_EXTRA_LINEAR = os.path.join(PATH_EXTRA, 'linear')
PATH_EXTRA_KNN = os.path.join(PATH_EXTRA, 'knn')
PATH_EXTRA_BIOINSP = os.path.join(PATH_EXTRA, 'bioinsp')

PATH_FIGURES = os.path.join(PATH_RESULTS, 'figures')

FILE_RESULTS = 'results_{}.json'

DATETIME_FORMAT = '%Y-%m-%d_%H;%M;%S'
DELIMITER = ';'

# error message due to data availability
MSSG_ERROR_DATA = 'The dataset which supports the findings of this study is available on request from the corresponding author.\n' \
                  'However, the human data could not be made publicly available due to restrictions from our Ethics Committees for Clinical Research, ' \
                  'as they contain information that might compromise the privacy of the patients in the cohort.\n' \
                  'Please contact mailto:fegarcia@bcamath.org'

# data
NUM_CLASSES = 3

with open(FILE_VAR_NAMES, 'r') as f_vars:
    VAR_NAMES = json.load(f_vars)

VAR_GROUP = 'hospital'
VAR_CLASSIF = 'severity_ordinal'
VARS_STRATIF = [VAR_GROUP, VAR_CLASSIF]

VARS_EXTRA = ['severity_binary', 'death']

# variable encoding
VARS_IN = [var for var in VAR_NAMES.values() if var not in VARS_STRATIF + VARS_EXTRA]

VARS_ORDINAL = {'pat_alcohol': 3, 'pat_tobacco': 3, 'neumo_curb65': 5, 'sepsis_qsofa': 4,
                'symptoms_fever': 3, 'covid-treatm_cortic-iv': 3, 'covid-treatm_lmwh': 5}

VARS_CATEGORICAL = {'comorb_broncho': ['No', 'Asthma', 'DPLP', 'COPD', 'Others'],  # never missing
                    'emerg-treatm_cortico': ['No', 'Inhaled', 'Oral', 'Missing'],
                    'emerg-pulmo_infiltr-xr': ['No', 'Unilobar', 'Multilob unilat', 'Bilateral', 'Missing'],
                    'emerg-pulmo_infiltr-type': ['No', 'Alveolar', 'Consolidation', 'Interstitial', 'Missing'],
                    'covid-diagn_method': ['Fast serology', 'PCR sputum', 'PCR nasophar', 'Missing'],
                    'covid-treatm_antibiot': ['No', 'Beta-lactam', 'Macrolides', 'Macrol & Beta', 'Quinolones', 'Others']}  # never missing

VARS_CONTINUOUS = ['pat_age', 'pat_height', 'pat_weight', 'pat_bmi', 'comorb_charlson', 'neumo_psi-sc',
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

# bootstrap and its random seed
N_BOOTSTR = 100

# - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * -
# - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * -
# - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * -
# ATTENTION: always fix the same seed, for reproducible bootstrap replicas!!!
# ATTENTION: always fix the same seed, for reproducible bootstrap replicas!!!
# ATTENTION: always fix the same seed, for reproducible bootstrap replicas!!!
SEED_BOOTSTRAP = 0
# ATTENTION: always fix the same seed, for reproducible bootstrap replicas!!!
# ATTENTION: always fix the same seed, for reproducible bootstrap replicas!!!
# ATTENTION: always fix the same seed, for reproducible bootstrap replicas!!!
# - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * -
# - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * -
# - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * -

# computation
N_CV_SPLITS = 3

# PARALLEL = False
PARALLEL = True

NUM_CORES = 8  # typical multicore processor
# NUM_CORES = 20  # cluster computing

# analysis of results
ALPHA = 0.05  # statistics

SIMILARITY_METHOD = 'jaccard'
STABIL_THRESH = 0.70
FREQ_FEATS_THRESH = 0.80
