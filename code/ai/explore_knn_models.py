import os
import json
import numpy as np
import pandas as pd

from numpy.random import SeedSequence

from bootstrap import bootstrap_classes, bootstrap_groups_classes

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

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
K_NEIGHB = [5, 10, 15, 20, 25]

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
    knnc = KNeighborsClassifier(weights='distance')
    knnr = KNeighborsRegressor(weights='distance')

    est_classif = Pipeline([('balancer', balancer), ('estimator', knnc)])
    est_regress = Pipeline([('balancer', balancer), ('estimator', knnr)])

    # grid search
    param_grid = {'estimator__n_neighbors': K_NEIGHB}

    search_classif = GridSearchCV(estimator=est_classif, param_grid=param_grid,
                                  scoring=scorer_classif, cv=StratifiedKFold(n_splits=config.N_CV_SPLITS),
                                  refit=False, n_jobs=-1, verbose=VERBOSE)
    search_regress = GridSearchCV(estimator=est_regress, param_grid=param_grid,
                                  scoring=scorer_regress, cv=StratifiedKFold(n_splits=config.N_CV_SPLITS),
                                  refit=False, n_jobs=-1, verbose=VERBOSE)

    # perform hyperparameter tuning searches
    search_classif.fit(X, Y)
    search_regress.fit(X, Y)

    # pack results
    k_opt_classif = search_classif.best_params_['estimator__n_neighbors']
    k_opt_regress = search_regress.best_params_['estimator__n_neighbors']

    score_classif = search_classif.best_score_
    score_regress = search_regress.best_score_

    results = {'classif': {'k_opt': k_opt_classif, 'gms': score_classif},
               'regress': {'k_opt': k_opt_regress, 'gms': score_regress}}

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
    idx_iter = pd.RangeIndex(start=0, stop=config.N_BOOTSTR)
    df_knnc = pd.DataFrame(index=idx_iter, columns=['k_opt', 'gms'])
    df_knnr = pd.DataFrame(index=idx_iter, columns=['k_opt', 'gms'])
    for idx, result in enumerate(results_hyperp):
        df_knnc.loc[idx, 'k_opt'] = result['classif']['k_opt']
        df_knnc.loc[idx, 'gms'] = result['classif']['gms']

        df_knnr.loc[idx, 'k_opt'] = result['regress']['k_opt']
        df_knnr.loc[idx, 'gms'] = result['regress']['gms']

    results = {'encoder': ENCODER, 'scaler': SCALER,
               'imputer': IMPUTER, 'imputer_params': IMPUTER_PARAMS,
               'balancer': BALANCER, 'balancer_params': BALANCER_PARAMS,
               'knnc': df_knnc.to_json(), 'knnr': df_knnr.to_json()}

    filename = config.FILE_RESULTS.format('explore_knn')
    f_results = os.path.join(config.PATH_EXTRA_KNN, filename)
    with open(f_results, 'w') as f_json_results:
        json.dump(results, f_json_results)
