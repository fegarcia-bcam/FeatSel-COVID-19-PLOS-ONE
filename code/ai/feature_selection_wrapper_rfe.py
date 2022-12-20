import os
import json
import pandas as pd
from datetime import datetime

from functools import partial
from numpy.random import SeedSequence
from pathos.pools import ProcessPool

from bootstrap import bootstrap_classes, bootstrap_groups_classes
import feature_selection_preproc
import feature_selection_classif
import feature_selection_regress
import config


# encoding
ENCODER = 'one-hot'

# scaling
SCALER = 'robust'

# imputation
IMPUTERS = [{'imputer': 'knn', 'imputer_params': {'add_indicator': False, 'n_neighbors': 9, 'weights': 'distance'}},
            {'imputer': 'iterative', 'imputer_params': {'add_indicator': False, 'n_nearest_features': 4,
                                                        'initial_strategy': 'median', 'sample_posterior': True}}]

# balancing
BALANCER = 'none'
BALANCER_PARAMS = {}

# feature selection
FEATURE_SELECTOR = 'wrap-rfe'

# feature selection parameters
STEP = 1

FEAT_SELECTORS = [{'analysis': 'classif', 'selector_params': {'estimator': 'LogReg', 'n_features_to_select': 5, 'step': STEP}},
                  {'analysis': 'classif', 'selector_params': {'estimator': 'LogReg', 'n_features_to_select': 10, 'step': STEP}},
                  {'analysis': 'classif', 'selector_params': {'estimator': 'LogReg', 'n_features_to_select': 20, 'step': STEP}},
                  {'analysis': 'classif', 'selector_params': {'estimator': 'LogReg', 'n_features_to_select': 40, 'step': STEP}},
                  {'analysis': 'regress', 'selector_params': {'estimator': 'Ridge', 'n_features_to_select': 5, 'step': STEP}},
                  {'analysis': 'regress', 'selector_params': {'estimator': 'Ridge', 'n_features_to_select': 10, 'step': STEP}},
                  {'analysis': 'regress', 'selector_params': {'estimator': 'Ridge', 'n_features_to_select': 20, 'step': STEP}},
                  {'analysis': 'regress', 'selector_params': {'estimator': 'Ridge', 'n_features_to_select': 40, 'step': STEP}}]


# pre-processing
def feat_preproc(df_in, encoder, scaler, imputer, imputer_params, balancer, balancer_params):
    df_X = df_in.drop(columns=config.VARS_EXTRA + config.VARS_STRATIF)
    df_Y = df_in[config.VAR_CLASSIF].astype('int')

    # pre-process data
    X, Y, feat_names_in = feature_selection_preproc.preprocess(df_X, df_Y,
                                                               encoder=encoder, scaler=scaler,
                                                               imputer=imputer, imputer_params=imputer_params,
                                                               balancer=balancer, balancer_params=balancer_params)
    df_out = pd.DataFrame(data=X, columns=feat_names_in)
    df_out.insert(0, config.VAR_CLASSIF, Y)

    return df_out


# feature selection
def feat_sel_classif(df_in,
                     encoder, scaler, imputer, imputer_params, balancer, balancer_params,
                     feature_selector, selector_params):
    # pre-process data
    df_out = feat_preproc(df_in, encoder, scaler, imputer, imputer_params, balancer, balancer_params)

    # perform feature selection
    idx, time = feature_selection_classif.select_features(df_out, feature_selector, selector_params)
    return idx, time


def feat_sel_regress(df_in,
                     encoder, scaler, imputer, imputer_params, balancer, balancer_params,
                     feature_selector, selector_params):
    # pre-process data
    df_out = feat_preproc(df_in, encoder, scaler, imputer, imputer_params, balancer, balancer_params)

    # perform feature selection
    idx, time = feature_selection_regress.select_features(df_out, feature_selector, selector_params)
    return idx, time


if __name__ == '__main__':
    # load data
    df_data = pd.read_csv(config.FILE_DATA_IN, sep=config.DELIMITER)
    if len(df_data.columns) == 1:
        raise SystemExit(config.MSSG_ERROR_DATA)

    # run bootstrap
    seed_seq = SeedSequence(config.SEED_BOOTSTRAP)
    # boost_list = bootstrap_classes(config.N_BOOTSTR, df_data, config.VAR_CLASSIF, seed_seq)
    boost_list = bootstrap_groups_classes(config.N_BOOTSTR, df_data, config.VAR_GROUP, config.VAR_CLASSIF, seed_seq)

    # feature selection

    for sel in FEAT_SELECTORS:
        analysis = sel['analysis']
        selector_params = sel['selector_params']

        if analysis == 'classif':
            for imp in IMPUTERS:
                imputer = imp['imputer']
                imputer_params = imp['imputer_params']

                if config.PARALLEL:
                    # parallelized
                    pool_bootstr = ProcessPool(nodes=config.NUM_CORES)
                    f_partial_c = partial(feat_sel_classif,
                                          encoder=ENCODER,
                                          scaler=SCALER,
                                          imputer=imputer, imputer_params=imputer_params,
                                          balancer=BALANCER, balancer_params=BALANCER_PARAMS,
                                          feature_selector=FEATURE_SELECTOR, selector_params=selector_params)
                    l_results_feat_sel = pool_bootstr.map(f_partial_c, boost_list)
                    # close parallel computation
                    pool_bootstr.close()
                    pool_bootstr.join()
                    pool_bootstr.clear()

                else:
                    # serial loop
                    l_results_feat_sel = []
                    for boost_item in boost_list:
                        result_feat_sel = feat_sel_classif(boost_item,
                                                           encoder=ENCODER,
                                                           scaler=SCALER,
                                                           imputer=imputer, imputer_params=imputer_params,
                                                           balancer=BALANCER, balancer_params=BALANCER_PARAMS,
                                                           feature_selector=FEATURE_SELECTOR, selector_params=selector_params)
                        l_results_feat_sel.append(result_feat_sel)

                # reorder results
                l_results_feat_sel = list(map(list, zip(*l_results_feat_sel)))
                selection, ttotal = l_results_feat_sel

                # save results
                results = {'analysis': analysis,
                           'encoder': ENCODER, 'scaler': SCALER,
                           'imputer': imputer, 'imputer_params': imputer_params,
                           'balancer': BALANCER, 'balancer_params': BALANCER_PARAMS,
                           'feature_selector': FEATURE_SELECTOR, 'selector_params': selector_params,
                           'sel_result': selection, 'time': ttotal}

                timestamp = datetime.now().strftime(config.DATETIME_FORMAT)
                filename = config.FILE_RESULTS.format(timestamp)
                f_results = os.path.join(config.PATH_RESULTS_WRA_RFE, filename)
                with open(f_results, 'w') as f_json_results:
                    json.dump(results, f_json_results)

        elif analysis == 'regress':
            for imp in IMPUTERS:
                imputer = imp['imputer']
                imputer_params = imp['imputer_params']

                if config.PARALLEL:
                    # parallelized
                    pool_bootstr = ProcessPool(nodes=config.NUM_CORES)
                    f_partial_r = partial(feat_sel_regress,
                                          encoder=ENCODER,
                                          scaler=SCALER,
                                          imputer=imputer, imputer_params=imputer_params,
                                          balancer=BALANCER, balancer_params=BALANCER_PARAMS,
                                          feature_selector=FEATURE_SELECTOR, selector_params=selector_params)
                    l_results_feat_sel = pool_bootstr.map(f_partial_r, boost_list)
                    # close parallel computation
                    pool_bootstr.close()
                    pool_bootstr.join()
                    pool_bootstr.clear()

                else:
                    # serial loop
                    l_results_feat_sel = []
                    for boost_item in boost_list:
                        result_feat_sel = feat_sel_regress(boost_item,
                                                           encoder=ENCODER,
                                                           scaler=SCALER,
                                                           imputer=imputer, imputer_params=imputer_params,
                                                           balancer=BALANCER, balancer_params=BALANCER_PARAMS,
                                                           feature_selector=FEATURE_SELECTOR, selector_params=selector_params)
                        l_results_feat_sel.append(result_feat_sel)

                # reorder results
                l_results_feat_sel = list(map(list, zip(*l_results_feat_sel)))
                selection, ttotal = l_results_feat_sel

                # save results
                results = {'analysis': analysis,
                           'encoder': ENCODER, 'scaler': SCALER,
                           'imputer': imputer, 'imputer_params': imputer_params,
                           'balancer': BALANCER, 'balancer_params': BALANCER_PARAMS,
                           'feature_selector': FEATURE_SELECTOR, 'selector_params': selector_params,
                           'sel_result': selection, 'time': ttotal}

                timestamp = datetime.now().strftime(config.DATETIME_FORMAT)
                filename = config.FILE_RESULTS.format(timestamp)
                f_results = os.path.join(config.PATH_RESULTS_WRA_RFE, filename)
                with open(f_results, 'w') as f_json_results:
                    json.dump(results, f_json_results)

        else:
            raise RuntimeError
