import copy
import numpy as np

from time import perf_counter

from sklearn.model_selection import StratifiedKFold

from sklearn.feature_selection import mutual_info_regression, GenericUnivariateSelect
from sklearn.feature_selection import SelectFromModel, RFE, RFECV, SequentialFeatureSelector

from sklearn.linear_model import Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

from imblearn.over_sampling import RandomOverSampler
from imblearn.metrics import geometric_mean_score

import config
from FeatSelPipeline import FeatSelPipeline
import bio_inspired_regress as bi


BALANCER_PIPE = RandomOverSampler(sampling_strategy='not majority')

A_RIDGE = 0.1
RIDGE = Ridge(alpha=A_RIDGE)
KNNR = KNeighborsRegressor(n_neighbors=5, weights='distance')
HGBR = HistGradientBoostingRegressor(max_bins=25, early_stopping=True, tol=1e-4, max_iter=20)

GAMMA_SEL = 0.8


def score_general(estimator, X, y, num_feats_total, alpha):
    num_feats = X.shape[1]
    y_pred = estimator.predict(X)
    y_pred = np.around(y_pred)
    y_pred[y_pred < 0] = 0
    y_pred[y_pred > config.NUM_CLASSES - 1] = config.NUM_CLASSES - 1

    score_clf = geometric_mean_score(y, y_pred)
    score = alpha * score_clf + (1.0 - alpha) * (1 - num_feats / num_feats_total)
    return score


def scorer_sfs(estimator, X, y):
    y_pred = estimator.predict(X)
    y_pred = np.around(y_pred)
    y_pred[y_pred < 0] = 0
    y_pred[y_pred > config.NUM_CLASSES - 1] = config.NUM_CLASSES - 1
    score = geometric_mean_score(y, y_pred)
    return score


def select_features(df, feature_selector, selector_params):
    sel_params = copy.deepcopy(selector_params)

    if feature_selector == 'filter-univar':
        if sel_params['score_func'] == 'mutual_info_regression':
            sel_params['score_func'] = mutual_info_regression
        selector = GenericUnivariateSelect(**sel_params)

        Y = df[config.VAR_CLASSIF]
        X = df.drop(config.VAR_CLASSIF, axis=1)
        n = X.shape[1]

        tic = perf_counter()
        sel = selector.fit(X, Y.astype('int'))
        toc = perf_counter()
        time = toc - tic

        idx = sel.get_support(indices=True)
        cols = np.zeros(n)
        cols[idx] = 1

    elif feature_selector == 'embed':
        sel_params['estimator'] = Lasso(max_iter=10000, alpha=selector_params['estimator'])
        selector = SelectFromModel(**sel_params)

        Y = df[config.VAR_CLASSIF]
        X = df.drop(config.VAR_CLASSIF, axis=1)
        n = X.shape[1]

        tic = perf_counter()
        sel = selector.fit(X, Y.astype('int'))
        toc = perf_counter()

        time = toc - tic
        idx = sel.get_support(indices=True)
        cols = np.zeros(n)
        cols[idx] = 1

    elif feature_selector == 'wrap-sfs':
        if sel_params['estimator'] == 'Ridge':
            sel_params['estimator'] = FeatSelPipeline([('balancer', BALANCER_PIPE), ('estimator', RIDGE)])
        elif sel_params['estimator'] == 'KNNR':
            sel_params['estimator'] = FeatSelPipeline([('balancer', BALANCER_PIPE), ('estimator', KNNR)])
        elif sel_params['estimator'] == 'HGBR':
            sel_params['estimator'] = FeatSelPipeline([('balancer', BALANCER_PIPE), ('estimator', HGBR)])
        else:
            raise NotImplementedError

        n_splits = sel_params['cv']
        sel_params['cv'] = StratifiedKFold(n_splits=n_splits)
        selector = SequentialFeatureSelector(**sel_params, scoring=scorer_sfs)

        Y = df[config.VAR_CLASSIF]
        X = df.drop(config.VAR_CLASSIF, axis=1)
        n = X.shape[1]

        tic = perf_counter()
        sel = selector.fit(X, Y.astype('int'))
        toc = perf_counter()

        time = toc - tic
        idx = sel.get_support(indices=True)
        cols = np.zeros(n)
        cols[idx] = 1

    elif feature_selector == 'wrap-rfe':
        if sel_params['estimator'] == 'Ridge':
            sel_params['estimator'] = FeatSelPipeline([('balancer', BALANCER_PIPE), ('estimator', RIDGE)])
        elif sel_params['estimator'] == 'HGBR':
            sel_params['estimator'] = FeatSelPipeline([('balancer', BALANCER_PIPE), ('estimator', HGBR)])
        else:
            raise NotImplementedError
        selector = RFE(**sel_params, importance_getter='named_steps.estimator.coef_')

        Y = df[config.VAR_CLASSIF]
        X = df.drop(config.VAR_CLASSIF, axis=1)
        num_feats_data = X.shape[1]

        tic = perf_counter()
        sel = selector.fit(X, Y.astype('int'))
        toc = perf_counter()

        time = toc - tic
        idx = sel.get_support(indices=True)
        cols = np.zeros(num_feats_data)
        cols[idx] = 1

    elif feature_selector == 'wrap-rfecv':
        if sel_params['estimator'] == 'Ridge':
            sel_params['estimator'] = FeatSelPipeline([('balancer', BALANCER_PIPE), ('estimator', RIDGE)])
        elif sel_params['estimator'] == 'HGBR':
            sel_params['estimator'] = FeatSelPipeline([('balancer', BALANCER_PIPE), ('estimator', HGBR)])
        else:
            raise NotImplementedError
        n_splits = sel_params['cv']
        sel_params['cv'] = StratifiedKFold(n_splits=n_splits)

        Y = df[config.VAR_CLASSIF]
        X = df.drop(config.VAR_CLASSIF, axis=1)
        num_feats_data = X.shape[1]

        def score_particular(estimator, X, y):
            return score_general(estimator, X, y, num_feats_total=num_feats_data, alpha=GAMMA_SEL)

        selector = RFECV(**sel_params, scoring=score_particular, importance_getter='named_steps.estimator.coef_')

        tic = perf_counter()
        sel = selector.fit(X, Y.astype('int'))
        toc = perf_counter()

        time = toc - tic
        idx = sel.get_support(indices=True)
        cols = np.zeros(num_feats_data)
        cols[idx] = 1

    elif feature_selector == 'wrap-ga':
        if sel_params['estimator'] == 'Ridge':
            sel_params['estimator'] = FeatSelPipeline([('balancer', BALANCER_PIPE), ('estimator', RIDGE)])
        elif sel_params['estimator'] == 'KNNR':
            sel_params['estimator'] = FeatSelPipeline([('balancer', BALANCER_PIPE), ('estimator', KNNR)])
        elif sel_params['estimator'] == 'HGBR':
            sel_params['estimator'] = FeatSelPipeline([('balancer', BALANCER_PIPE), ('estimator', HGBR)])
        else:
            raise NotImplementedError

        Y = df[config.VAR_CLASSIF]
        X = df.drop(config.VAR_CLASSIF, axis=1)

        tic = perf_counter()
        X_new, cols = bi.ga(X.to_numpy(), Y, **sel_params)
        toc = perf_counter()
        time = toc - tic

    elif feature_selector == 'wrap-bpso':
        if sel_params['estimator'] == 'Ridge':
            sel_params['estimator'] = FeatSelPipeline([('balancer', BALANCER_PIPE), ('estimator', RIDGE)])
        elif sel_params['estimator'] == 'KNNR':
            sel_params['estimator'] = FeatSelPipeline([('balancer', BALANCER_PIPE), ('estimator', KNNR)])
        elif sel_params['estimator'] == 'HGBR':
            sel_params['estimator'] = FeatSelPipeline([('balancer', BALANCER_PIPE), ('estimator', HGBR)])
        else:
            raise NotImplementedError

        Y = df[config.VAR_CLASSIF]
        X = df.drop(config.VAR_CLASSIF, axis=1)

        tic = perf_counter()
        X_new, cols = bi.bpso(X.to_numpy(), Y, **sel_params)
        toc = perf_counter()

        time = toc - tic

    else:
        raise NotImplementedError

    return cols.tolist(), time
