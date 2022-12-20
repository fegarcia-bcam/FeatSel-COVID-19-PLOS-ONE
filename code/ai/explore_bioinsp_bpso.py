import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

from functools import partial
from numpy.random import SeedSequence
from pathos.pools import ProcessPool

import pyswarms as ps

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from imblearn.over_sampling import RandomOverSampler
from imblearn.metrics import geometric_mean_score

from bootstrap import bootstrap_classes, bootstrap_groups_classes
import feature_selection_preproc

from FeatSelPipeline import FeatSelPipeline

import config


N_BOOTSTR_EXPL = 10

BALANCER_PIPE = RandomOverSampler(sampling_strategy='not majority', shrinkage=None)

C_LR = 0.001
LOGREG = LogisticRegression(penalty='l2', solver='saga', C=C_LR, multi_class='multinomial', max_iter=10000)
HGBC = HistGradientBoostingClassifier(loss='log_loss', max_bins=50, early_stopping=True)

A_RIDGE = 0.1
RIDGE = Ridge(alpha=A_RIDGE)
HGBR = HistGradientBoostingRegressor(max_bins=50, early_stopping=True)


# encoding
ENCODER = 'one-hot'

# scaling
SCALER = 'robust'

# imputation
# different alternatives

# balancing
BALANCER = 'none'
BALANCER_PARAMS = {}

# feature selection
FEATURE_SELECTOR = 'wrap-bpso'

# feature selection parameters
GAMMA_SEL = 0.8
N_PARTICLES = 30
ITERS_MAIN = 2000
ITERS_HGB = 1000

OPTS_BPSO_A = {'c1': 0.5, 'c2': 0.5, 'w': 0.6, 'k': 30, 'p': 1}
OPTS_BPSO_B = {'c1': 0.5, 'c2': 0.5, 'w': 0.9, 'k': 30, 'p': 1}

FEAT_SELECTORS_MAIN = [{'analysis': 'classif',
                        'imputer': 'knn', 'imputer_params': {'add_indicator': False, 'n_neighbors': 9, 'weights': 'distance'},
                        'selector_params': {'gamma': GAMMA_SEL, 'n_sp': config.N_CV_SPLITS,
                                            'estimator': 'LogReg',
                                            'n_particles': N_PARTICLES, 'iters': ITERS_MAIN, 'options': OPTS_BPSO_A, 'velocity_clamp': (-2, 2)}},
                       {'analysis': 'classif',
                        'imputer': 'knn', 'imputer_params': {'add_indicator': False, 'n_neighbors': 9, 'weights': 'distance'},
                        'selector_params': {'gamma': GAMMA_SEL, 'n_sp': config.N_CV_SPLITS,
                                            'estimator': 'LogReg',
                                            'n_particles': N_PARTICLES, 'iters': ITERS_MAIN, 'options': OPTS_BPSO_B, 'velocity_clamp': (-2, 2)}},
                       {'analysis': 'classif',
                        'imputer': 'knn', 'imputer_params': {'add_indicator': False, 'n_neighbors': 9, 'weights': 'distance'},
                        'selector_params': {'gamma': GAMMA_SEL, 'n_sp': config.N_CV_SPLITS,
                                            'estimator': 'LogReg',
                                            'n_particles': N_PARTICLES, 'iters': ITERS_MAIN, 'options': OPTS_BPSO_A, 'velocity_clamp': (-6, 6)}},
                       {'analysis': 'classif',
                        'imputer': 'knn', 'imputer_params': {'add_indicator': False, 'n_neighbors': 9, 'weights': 'distance'},
                        'selector_params': {'gamma': GAMMA_SEL, 'n_sp': config.N_CV_SPLITS,
                                            'estimator': 'LogReg',
                                            'n_particles': N_PARTICLES, 'iters': ITERS_MAIN, 'options': OPTS_BPSO_B, 'velocity_clamp': (-6, 6)}},
                       {'analysis': 'regress',
                        'imputer': 'knn', 'imputer_params': {'add_indicator': False, 'n_neighbors': 9, 'weights': 'distance'},
                        'selector_params': {'gamma': GAMMA_SEL, 'n_sp': config.N_CV_SPLITS,
                                            'estimator': 'Ridge',
                                            'n_particles': N_PARTICLES, 'iters': ITERS_MAIN, 'options': OPTS_BPSO_A, 'velocity_clamp': (-2, 2)}},
                       {'analysis': 'regress',
                        'imputer': 'knn', 'imputer_params': {'add_indicator': False, 'n_neighbors': 9, 'weights': 'distance'},
                        'selector_params': {'gamma': GAMMA_SEL, 'n_sp': config.N_CV_SPLITS,
                                            'estimator': 'Ridge',
                                            'n_particles': N_PARTICLES, 'iters': ITERS_MAIN, 'options': OPTS_BPSO_B, 'velocity_clamp': (-2, 2)}},
                       {'analysis': 'regress',
                        'imputer': 'knn', 'imputer_params': {'add_indicator': False, 'n_neighbors': 9, 'weights': 'distance'},
                        'selector_params': {'gamma': GAMMA_SEL, 'n_sp': config.N_CV_SPLITS,
                                            'estimator': 'Ridge',
                                            'n_particles': N_PARTICLES, 'iters': ITERS_MAIN, 'options': OPTS_BPSO_A, 'velocity_clamp': (-6, 6)}},
                       {'analysis': 'regress',
                        'imputer': 'knn', 'imputer_params': {'add_indicator': False, 'n_neighbors': 9, 'weights': 'distance'},
                        'selector_params': {'gamma': GAMMA_SEL, 'n_sp': config.N_CV_SPLITS,
                                            'estimator': 'Ridge',
                                            'n_particles': N_PARTICLES, 'iters': ITERS_MAIN, 'options': OPTS_BPSO_B, 'velocity_clamp': (-6, 6)}}]

FEAT_SELECTORS_HGB = [{'analysis': 'classif',
                       'imputer': 'none', 'imputer_params': {},
                       'selector_params': {'gamma': GAMMA_SEL, 'n_sp': config.N_CV_SPLITS,
                                           'estimator': 'HGBC',
                                           'n_particles': N_PARTICLES, 'iters': ITERS_HGB, 'options': OPTS_BPSO_A, 'velocity_clamp': (-2, 2)}},
                      {'analysis': 'classif',
                       'imputer': 'none', 'imputer_params': {},
                       'selector_params': {'gamma': GAMMA_SEL, 'n_sp': config.N_CV_SPLITS,
                                           'estimator': 'HGBC',
                                           'n_particles': N_PARTICLES, 'iters': ITERS_HGB, 'options': OPTS_BPSO_B, 'velocity_clamp': (-2, 2)}},
                      {'analysis': 'classif',
                       'imputer': 'none', 'imputer_params': {},
                       'selector_params': {'gamma': GAMMA_SEL, 'n_sp': config.N_CV_SPLITS,
                                           'estimator': 'HGBC',
                                           'n_particles': N_PARTICLES, 'iters': ITERS_HGB, 'options': OPTS_BPSO_A, 'velocity_clamp': (-6, 6)}},
                      {'analysis': 'classif',
                       'imputer': 'none', 'imputer_params': {},
                       'selector_params': {'gamma': GAMMA_SEL, 'n_sp': config.N_CV_SPLITS,
                                           'estimator': 'HGBC',
                                           'n_particles': N_PARTICLES, 'iters': ITERS_HGB, 'options': OPTS_BPSO_B, 'velocity_clamp': (-6, 6)}},
                      {'analysis': 'regress',
                       'imputer': 'none', 'imputer_params': {},
                       'selector_params': {'gamma': GAMMA_SEL, 'n_sp': config.N_CV_SPLITS,
                                           'estimator': 'HGBR',
                                           'n_particles': N_PARTICLES, 'iters': ITERS_HGB, 'options': OPTS_BPSO_A, 'velocity_clamp': (-2, 2)}},
                      {'analysis': 'regress',
                       'imputer': 'none', 'imputer_params': {},
                       'selector_params': {'gamma': GAMMA_SEL, 'n_sp': config.N_CV_SPLITS,
                                           'estimator': 'HGBR',
                                           'n_particles': N_PARTICLES, 'iters': ITERS_HGB, 'options': OPTS_BPSO_B, 'velocity_clamp': (-2, 2)}},
                      {'analysis': 'regress',
                       'imputer': 'none', 'imputer_params': {},
                       'selector_params': {'gamma': GAMMA_SEL, 'n_sp': config.N_CV_SPLITS,
                                           'estimator': 'HGBR',
                                           'n_particles': N_PARTICLES, 'iters': ITERS_HGB, 'options': OPTS_BPSO_A, 'velocity_clamp': (-6, 6)}},
                      {'analysis': 'regress',
                       'imputer': 'none', 'imputer_params': {},
                       'selector_params': {'gamma': GAMMA_SEL, 'n_sp': config.N_CV_SPLITS,
                                           'estimator': 'HGBR',
                                           'n_particles': N_PARTICLES, 'iters': ITERS_HGB, 'options': OPTS_BPSO_B, 'velocity_clamp': (-6, 6)}}]


def bpso_classif(df_data,
                 encoder, scaler, imputer, imputer_params, balancer, balancer_params,
                 selector_params):
    df_X = df_data.drop(columns=config.VARS_EXTRA + config.VARS_STRATIF)
    df_Y = df_data[config.VAR_CLASSIF].astype('int')

    # pre-process data
    X, Y, feat_names_in = feature_selection_preproc.preprocess(df_X, df_Y,
                                                               encoder=encoder, scaler=scaler,
                                                               imputer=imputer, imputer_params=imputer_params,
                                                               balancer=balancer, balancer_params=balancer_params)

    # run
    f = swarm_classif(X, Y, **selector_params)
    return f


def bpso_regress(df_data,
                 encoder, scaler, imputer, imputer_params, balancer, balancer_params,
                 selector_params):
    df_X = df_data.drop(columns=config.VARS_EXTRA + config.VARS_STRATIF)
    df_Y = df_data[config.VAR_CLASSIF].astype('int')

    # pre-process data
    X, Y, feat_names_in = feature_selection_preproc.preprocess(df_X, df_Y,
                                                               encoder=encoder, scaler=scaler,
                                                               imputer=imputer, imputer_params=imputer_params,
                                                               balancer=balancer, balancer_params=balancer_params)

    # run
    f = swarm_regress(X, Y, **selector_params)
    return f


def swarm_classif(X, Y, alp, n_sp, estimator, n_particles, iters, options, velocity_clamp):
    dim = np.shape(X)[1]

    def f_aux(m, alpha, nsplits):
        n_feats_tot = dim
        if np.count_nonzero(m) == 0:
            X_subset = X
        else:
            X_subset = X[:, m == 1]
        cv_set = np.repeat(-1., X.shape[0])
        n_feats = X_subset.shape[1]

        skf = StratifiedKFold(n_splits=nsplits, shuffle=True)
        for train_index, test_index in skf.split(X_subset, Y):
            x_train, x_test = X_subset[train_index], X_subset[test_index]
            y_train, y_test = Y[train_index], Y[test_index]
            if x_train.shape[0] != y_train.shape[0]:
                raise Exception()

            est = clone(estimator)
            est.fit(x_train, y_train)
            y_pred = est.predict(x_test)
            cv_set[test_index] = y_pred

        sc_classif = geometric_mean_score(Y, cv_set)
        j = -(alpha * sc_classif + (1.0 - alpha) * (1 - (n_feats / n_feats_tot)))
        return j

    def f(x, alpha=alp, nsplits=n_sp):
        j = [f_aux(m=part, alpha=alpha, nsplits=nsplits) for part in x]
        return np.array(j)

    if estimator == 'LogReg':
        estimator = FeatSelPipeline([('balancer', BALANCER_PIPE), ('estimator', LOGREG)])
    elif estimator == 'HGBC':
        estimator = FeatSelPipeline([('balancer', BALANCER_PIPE), ('estimator', HGBC)])
    else:
        raise NotImplementedError

    optimizer = ps.discrete.BinaryPSO(n_particles=n_particles, dimensions=dim, options=options, velocity_clamp=velocity_clamp)
    cost, pos = optimizer.optimize(f, iters=iters)

    fitness = optimizer.cost_history
    return fitness


def swarm_regress(X, Y, alp, n_sp, estimator, n_particles, iters, options, velocity_clamp):
    dim = np.shape(X)[1]

    def f_aux(m, alpha, nsplits):
        n_feats_tot = dim
        if np.count_nonzero(m) == 0:
            X_subset = X
        else:
            X_subset = X[:, m == 1]
        cv_set = np.repeat(-1., X.shape[0])
        n_feats = X_subset.shape[1]

        skf = StratifiedKFold(n_splits=nsplits, shuffle=True)
        for train_index, test_index in skf.split(X_subset, Y):
            x_train, x_test = X_subset[train_index], X_subset[test_index]
            y_train, y_test = Y[train_index], Y[test_index]
            if x_train.shape[0] != y_train.shape[0]:
                raise Exception()

            mod = clone(estimator)
            mod.fit(x_train, y_train)
            y_pred = mod.predict(x_test)
            cv_set[test_index] = y_pred

        y_pred = np.around(cv_set)
        y_pred[y_pred < 0] = 0
        y_pred[y_pred > config.NUM_CLASSES - 1] = config.NUM_CLASSES - 1
        sc_classif = geometric_mean_score(Y, y_pred)
        j = -(alpha * sc_classif + (1.0 - alpha) * (1 - (n_feats / n_feats_tot)))
        return j

    def f(x, alpha=alp, nsplits=n_sp):
        j = [f_aux(m=part, alpha=alpha, nsplits=nsplits) for part in x]
        return np.array(j)

    if estimator == 'Ridge':
        estimator = FeatSelPipeline([('balancer', BALANCER_PIPE), ('estimator', RIDGE)])
    elif estimator == 'HGBR':
        estimator = FeatSelPipeline([('balancer', BALANCER_PIPE), ('estimator', HGBR)])
    else:
        raise NotImplementedError

    optimizer = ps.discrete.BinaryPSO(n_particles=n_particles, dimensions=dim, options=options, velocity_clamp=velocity_clamp)
    cost, pos = optimizer.optimize(f, iters=iters)

    fitness = optimizer.cost_history
    return fitness


if __name__ == '__main__':
    # load data
    df_data = pd.read_csv(config.FILE_DATA_IN, sep=config.DELIMITER)
    if len(df_data.columns) == 1:
        raise SystemExit(config.MSSG_ERROR_DATA)

    # run bootstrap
    seed_seq = SeedSequence(config.SEED_BOOTSTRAP)
    # boost_list = bootstrap_classes(N_BOOTSTR_EXPL, df_data, config.VAR_CLASSIF, seed_seq)
    boost_list = bootstrap_groups_classes(N_BOOTSTR_EXPL, df_data, config.VAR_GROUP, config.VAR_CLASSIF, seed_seq)

    for sel in FEAT_SELECTORS_MAIN:
        analysis = sel['analysis']
        imputer = sel['imputer']
        imputer_params = sel['imputer_params']
        selector_params = sel['selector_params']

        if analysis == 'classif':
            if config.PARALLEL:
                # parallelized
                pool_bootstr = ProcessPool(nodes=config.NUM_CORES)
                f_partial_c = partial(bpso_classif,
                                      encoder=ENCODER,
                                      scaler=SCALER,
                                      imputer=imputer, imputer_params=imputer_params,
                                      balancer=BALANCER, balancer_params=BALANCER_PARAMS,
                                      selector_params=selector_params)
                l_results_feat_sel = pool_bootstr.map(f_partial_c, boost_list)
                # close parallel computation
                pool_bootstr.close()
                pool_bootstr.join()
                pool_bootstr.clear()

            else:
                # serial loop
                l_results_feat_sel = []
                for boost_item in boost_list:
                    result_feat_sel = bpso_classif(boost_item,
                                                   encoder=ENCODER,
                                                   scaler=SCALER,
                                                   imputer=imputer, imputer_params=imputer_params,
                                                   balancer=BALANCER, balancer_params=BALANCER_PARAMS,
                                                   selector_params=selector_params)
                    l_results_feat_sel.append(result_feat_sel)

            # reorder results
            l_results_feat_sel = list(map(list, zip(*l_results_feat_sel)))
            fitness = l_results_feat_sel

            # save results
            results = {'analysis': analysis,
                       'encoder': ENCODER, 'scaler': SCALER,
                       'imputer': imputer, 'imputer_params': imputer_params,
                       'balancer': BALANCER, 'balancer_params': BALANCER_PARAMS,
                       'feature_selector': FEATURE_SELECTOR, 'selector_params': selector_params,
                       'fitness': fitness}

            timestamp = datetime.now().strftime(config.DATETIME_FORMAT)
            filename = config.FILE_RESULTS.format(timestamp)
            f_results = os.path.join(config.PATH_EXTRA_BIOINSP, filename)

            with open(f_results, 'w') as f_json_results:
                json.dump(results, f_json_results)

        elif analysis == 'regress':
            if config.PARALLEL:
                # parallelized
                pool_bootstr = ProcessPool(nodes=config.NUM_CORES)
                f_partial_c = partial(bpso_regress,
                                      encoder=ENCODER,
                                      scaler=SCALER,
                                      imputer=imputer, imputer_params=imputer_params,
                                      balancer=BALANCER, balancer_params=BALANCER_PARAMS,
                                      selector_params=selector_params)
                l_results_feat_sel = pool_bootstr.map(f_partial_c, boost_list)
                # close parallel computation
                pool_bootstr.close()
                pool_bootstr.join()
                pool_bootstr.clear()

            else:
                # serial loop
                l_results_feat_sel = []
                for boost_item in boost_list:
                    result_feat_sel = bpso_regress(boost_item,
                                                   encoder=ENCODER,
                                                   scaler=SCALER,
                                                   imputer=imputer, imputer_params=imputer_params,
                                                   balancer=BALANCER, balancer_params=BALANCER_PARAMS,
                                                   selector_params=selector_params)
                    l_results_feat_sel.append(result_feat_sel)

            # reorder results
            l_results_feat_sel = list(map(list, zip(*l_results_feat_sel)))
            fitness = l_results_feat_sel

            # save results
            results = {'analysis': analysis,
                       'encoder': ENCODER, 'scaler': SCALER,
                       'imputer': imputer, 'imputer_params': imputer_params,
                       'balancer': BALANCER, 'balancer_params': BALANCER_PARAMS,
                       'feature_selector': FEATURE_SELECTOR, 'selector_params': selector_params,
                       'fitness': fitness}

            timestamp = datetime.now().strftime(config.DATETIME_FORMAT)
            filename = config.FILE_RESULTS.format(timestamp)
            f_results = os.path.join(config.PATH_EXTRA_BIOINSP, filename)

            with open(f_results, 'w') as f_json_results:
                json.dump(results, f_json_results)

        else:
            raise RuntimeError

    for sel in FEAT_SELECTORS_HGB:
        analysis = sel['analysis']
        imputer = sel['imputer']
        imputer_params = sel['imputer_params']
        selector_params = sel['selector_params']

        if analysis == 'classif':
            if config.PARALLEL:
                # parallelized
                pool_bootstr = ProcessPool(nodes=config.NUM_CORES)
                f_partial_c = partial(bpso_classif,
                                      encoder=ENCODER,
                                      scaler=SCALER,
                                      imputer=imputer, imputer_params=imputer_params,
                                      balancer=BALANCER, balancer_params=BALANCER_PARAMS,
                                      selector_params=selector_params)
                l_results_feat_sel = pool_bootstr.map(f_partial_c, boost_list)
                # close parallel computation
                pool_bootstr.close()
                pool_bootstr.join()
                pool_bootstr.clear()

            else:
                # serial loop
                l_results_feat_sel = []
                for boost_item in boost_list:
                    result_feat_sel = bpso_classif(boost_item,
                                                   encoder=ENCODER,
                                                   scaler=SCALER,
                                                   imputer=imputer, imputer_params=imputer_params,
                                                   balancer=BALANCER, balancer_params=BALANCER_PARAMS,
                                                   selector_params=selector_params)
                    l_results_feat_sel.append(result_feat_sel)

            # reorder results
            l_results_feat_sel = list(map(list, zip(*l_results_feat_sel)))
            fitness = l_results_feat_sel

            # save results
            results = {'analysis': analysis,
                       'encoder': ENCODER, 'scaler': SCALER,
                       'imputer': imputer, 'imputer_params': imputer_params,
                       'balancer': BALANCER, 'balancer_params': BALANCER_PARAMS,
                       'feature_selector': FEATURE_SELECTOR, 'selector_params': selector_params,
                       'fitness': fitness}

            timestamp = datetime.now().strftime(config.DATETIME_FORMAT)
            filename = config.FILE_RESULTS.format(timestamp)
            f_results = os.path.join(config.PATH_EXTRA_BIOINSP, filename)

            with open(f_results, 'w') as f_json_results:
                json.dump(results, f_json_results)

        elif analysis == 'regress':
            if config.PARALLEL:
                # parallelized
                pool_bootstr = ProcessPool(nodes=config.NUM_CORES)
                f_partial_c = partial(bpso_regress,
                                      encoder=ENCODER,
                                      scaler=SCALER,
                                      imputer=imputer, imputer_params=imputer_params,
                                      balancer=BALANCER, balancer_params=BALANCER_PARAMS,
                                      selector_params=selector_params)
                l_results_feat_sel = pool_bootstr.map(f_partial_c, boost_list)
                # close parallel computation
                pool_bootstr.close()
                pool_bootstr.join()
                pool_bootstr.clear()

            else:
                # serial loop
                l_results_feat_sel = []
                for boost_item in boost_list:
                    result_feat_sel = bpso_regress(boost_item,
                                                   encoder=ENCODER,
                                                   scaler=SCALER,
                                                   imputer=imputer, imputer_params=imputer_params,
                                                   balancer=BALANCER, balancer_params=BALANCER_PARAMS,
                                                   selector_params=selector_params)
                    l_results_feat_sel.append(result_feat_sel)

            # reorder results
            l_results_feat_sel = list(map(list, zip(*l_results_feat_sel)))
            fitness = l_results_feat_sel

            # save results
            results = {'analysis': analysis,
                       'encoder': ENCODER, 'scaler': SCALER,
                       'imputer': imputer, 'imputer_params': imputer_params,
                       'balancer': BALANCER, 'balancer_params': BALANCER_PARAMS,
                       'feature_selector': FEATURE_SELECTOR, 'selector_params': selector_params,
                       'fitness': fitness}

            timestamp = datetime.now().strftime(config.DATETIME_FORMAT)
            filename = config.FILE_RESULTS.format(timestamp)
            f_results = os.path.join(config.PATH_EXTRA_BIOINSP, filename)

            with open(f_results, 'w') as f_json_results:
                json.dump(results, f_json_results)

        else:
            raise RuntimeError
