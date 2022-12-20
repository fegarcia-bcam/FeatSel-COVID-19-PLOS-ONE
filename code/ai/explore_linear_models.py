import os
import json
import numpy as np
import pandas as pd

from numpy.random import SeedSequence

from bootstrap import bootstrap_classes, bootstrap_groups_classes

from skopt import BayesSearchCV
from skopt.space import Real

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression, Ridge

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from imblearn.metrics import geometric_mean_score

import feature_selection_preproc
import config

# encoding
ENCODER = 'one-hot'

# scaling
SCALER = 'robust'

# imputation
IMPUTER = 'knn'
IMPUTER_PARAMS = {'add_indicator': False, 'n_neighbors': 9, 'weights': 'distance'}

# balancing
BALANCER = 'none'
BALANCER_PARAMS = {}

# hyperparameter search range
LOG_BASE = 10.0

C_LOG_MIN = -5
C_LOG_MAX = +5  # scikit-learn's default log-range is: -/+ 4
C_NUM_ITER = 15  # scikit-learn's default is: 10

A_LOG_MIN = -2
A_LOG_MAX = +2  # scikit-learn's default log-range is: -/+ 1
A_NUM_ITER = 25  # scikit-learn's default is: 3, but with an efficient Leave-One-Out Cross-Validation

# verbosity
VERBOSE = False  # boolean or integer


def scorer_classif(estimator, X, y):
    y_pred = estimator.predict(X)
    score = geometric_mean_score(y, y_pred)
    return score


def scorer_regress(estimator, X, y):
    y_pred = estimator.predict(X)
    y_pred = np.around(y_pred)
    y_pred[y_pred < 0] = 0
    y_pred[y_pred > config.NUM_CLASSES - 1] = config.NUM_CLASSES - 1
    score = geometric_mean_score(y, y_pred)
    return score


def explore_hyperparam(df_in):
    df_X = df_in.drop(columns=config.VARS_EXTRA + config.VARS_STRATIF)
    df_Y = df_in[config.VAR_CLASSIF].astype('int')

    # pre-process data
    X, Y, feat_names_in = feature_selection_preproc.preprocess(df_X, df_Y,
                                                               ENCODER, SCALER,
                                                               IMPUTER, IMPUTER_PARAMS,
                                                               BALANCER, BALANCER_PARAMS)

    # within-pipe balancer, as there wasn't any during pre-processing
    balancer = RandomOverSampler(sampling_strategy='not majority', shrinkage=0.01)

    # linear models in a balancing pipeline
    logreg = LogisticRegression(penalty='l2', solver='lbfgs', multi_class='multinomial', max_iter=10000)
    ridge = Ridge()

    est_classif = Pipeline([('balancer', balancer), ('estimator', logreg)])
    est_regress = Pipeline([('balancer', balancer), ('estimator', ridge)])

    # grid search
    param_grid_c = {'estimator__C': np.logspace(start=C_LOG_MIN, stop=C_LOG_MAX,
                                                num=C_NUM_ITER, base=LOG_BASE, endpoint=True)}
    param_grid_a = {'estimator__alpha': np.logspace(start=A_LOG_MIN, stop=A_LOG_MAX,
                                                    num=A_NUM_ITER, base=LOG_BASE, endpoint=True)}
    search_c_grid = GridSearchCV(estimator=est_classif, param_grid=param_grid_c,
                                 scoring=scorer_classif, cv=StratifiedKFold(n_splits=config.N_CV_SPLITS),
                                 refit=False, n_jobs=-1, verbose=VERBOSE)
    search_a_grid = GridSearchCV(estimator=est_regress, param_grid=param_grid_a,
                                 scoring=scorer_regress, cv=StratifiedKFold(n_splits=config.N_CV_SPLITS),
                                 refit=False, n_jobs=-1, verbose=VERBOSE)

    # Bayesian search
    param_space_c = {'estimator__C': Real(low=LOG_BASE ** C_LOG_MIN, high=LOG_BASE ** C_LOG_MAX,
                                          prior='log-uniform')}
    param_space_a = {'estimator__alpha': Real(low=LOG_BASE ** A_LOG_MIN, high=LOG_BASE ** A_LOG_MAX,
                                              prior='log-uniform')}
    search_c_bayes = BayesSearchCV(estimator=est_classif, search_spaces=param_space_c, n_iter=C_NUM_ITER,
                                   scoring=scorer_classif, cv=StratifiedKFold(n_splits=config.N_CV_SPLITS),
                                   refit=False, n_jobs=-1, verbose=VERBOSE)
    search_a_bayes = BayesSearchCV(estimator=est_regress, search_spaces=param_space_a, n_iter=A_NUM_ITER,
                                   scoring=scorer_regress, cv=StratifiedKFold(n_splits=config.N_CV_SPLITS),
                                   refit=False, n_jobs=-1, verbose=VERBOSE)

    # perform hyperparameter tuning searches
    search_c_grid.fit(X, Y)
    search_a_grid.fit(X, Y)
    search_c_bayes.fit(X, Y)
    search_a_bayes.fit(X, Y)

    # pack results
    c_opt_grid = search_c_grid.best_params_['estimator__C']
    a_opt_grid = search_a_grid.best_params_['estimator__alpha']
    c_opt_bayes = search_c_bayes.best_params_['estimator__C']
    a_opt_bayes = search_a_bayes.best_params_['estimator__alpha']

    score_c_grid = search_c_grid.best_score_
    score_a_grid = search_a_grid.best_score_
    score_c_bayes = search_c_bayes.best_score_
    score_a_bayes = search_a_bayes.best_score_

    results = {'classif': {'grid': {'c_opt': c_opt_grid, 'gms': score_c_grid},
                           'bayes': {'c_opt': c_opt_bayes, 'gms': score_c_bayes}},
               'regress': {'grid': {'a_opt': a_opt_grid, 'gms': score_a_grid},
                           'bayes': {'a_opt': a_opt_bayes, 'gms': score_a_bayes}}}

    return results


if __name__ == '__main__':
    # load data
    df_data = pd.read_csv(config.FILE_DATA_IN, sep=config.DELIMITER)
    if len(df_data.columns) == 1:
        raise SystemExit(config.MSSG_ERROR_DATA)

    # run bootstrap
    seed_seq = SeedSequence(config.SEED_BOOTSTRAP)
    # boost_list = bootstrap_classes(config.N_BOOTSTR, df_data, config.VAR_CLASSIF, seed_seq)
    boost_list = bootstrap_groups_classes(config.N_BOOTSTR, df_data, config.VAR_GROUP, config.VAR_CLASSIF, seed_seq)

    # compute
    results_hyperp = []
    for boost_item in boost_list:
        result = explore_hyperparam(boost_item)
        results_hyperp.append(result)

    # store results
    idx_cols_c = pd.MultiIndex.from_product([['grid', 'bayes'], ['c', 'gms']])
    idx_cols_a = pd.MultiIndex.from_product([['grid', 'bayes'], ['a', 'gms']])
    idx_iter = pd.RangeIndex(start=0, stop=config.N_BOOTSTR)
    df_c = pd.DataFrame(index=idx_iter, columns=idx_cols_c)
    df_a = pd.DataFrame(index=idx_iter, columns=idx_cols_a)
    for idx, result in enumerate(results_hyperp):
        df_c.loc[idx, ('grid', 'c')] = result['classif']['grid']['c_opt']
        df_c.loc[idx, ('grid', 'gms')] = result['classif']['grid']['gms']
        df_c.loc[idx, ('bayes', 'c')] = result['classif']['bayes']['c_opt']
        df_c.loc[idx, ('bayes', 'gms')] = result['classif']['bayes']['gms']

        df_a.loc[idx, ('grid', 'a')] = result['regress']['grid']['a_opt']
        df_a.loc[idx, ('grid', 'gms')] = result['regress']['grid']['gms']
        df_a.loc[idx, ('bayes', 'a')] = result['regress']['bayes']['a_opt']
        df_a.loc[idx, ('bayes', 'gms')] = result['regress']['bayes']['gms']

    results = {'encoder': ENCODER, 'scaler': SCALER,
               'imputer': IMPUTER, 'imputer_params': IMPUTER_PARAMS,
               'balancer': BALANCER, 'balancer_params': BALANCER_PARAMS,
               'c': df_c.to_json(), 'a': df_a.to_json()}

    filename = config.FILE_RESULTS.format('explore_linear')
    f_results = os.path.join(config.PATH_EXTRA_LINEAR, filename)
    with open(f_results, 'w') as f_json_results:
        json.dump(results, f_json_results)
