import copy
import numpy as np

from time import perf_counter

from sklearn.model_selection import StratifiedKFold

from sklearn.feature_selection import mutual_info_classif, GenericUnivariateSelect
from sklearn.feature_selection import SelectFromModel, RFE, RFECV, SequentialFeatureSelector

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

from imblearn.over_sampling import RandomOverSampler
from imblearn.metrics import geometric_mean_score

import pymrmr
from skrebate import ReliefF, MultiSURF

import config
from FeatSelPipeline import FeatSelPipeline
import bio_inspired_classif as bi

from discretization import discretize
from fast_correlation import fcbf


BALANCER_PIPE = RandomOverSampler(sampling_strategy='not majority')

C_LR = 0.001
LOGREG = LogisticRegression(penalty='l2', solver='saga', C=C_LR, multi_class='multinomial', max_iter=10000)
KNNC = KNeighborsClassifier(n_neighbors=5, weights='distance')
HGBC = HistGradientBoostingClassifier(loss='log_loss', max_bins=25,
                                      early_stopping=True, tol=1e-4, max_iter=20)

GAMMA_SEL = 0.8


def score_general(estimator, X, y, num_feats_total, alpha):
    num_feats = X.shape[1]
    y_pred = estimator.predict(X)

    score_clf = geometric_mean_score(y, y_pred)
    score = alpha * score_clf + (1.0 - alpha) * (1 - num_feats / num_feats_total)
    return score


def scorer_sfs(estimator, X, y):
    y_pred = estimator.predict(X)
    score = geometric_mean_score(y, y_pred)
    return score


def select_features(df, feature_selector, selector_params):
    sel_params = copy.deepcopy(selector_params)

    if feature_selector == 'filter-univar':
        if sel_params['score_func'] == 'mutual_info_classif':
            sel_params['score_func'] = mutual_info_classif
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

    elif feature_selector == 'filt-mrmr':
        df_discr, _ = discretize(df, leave_continuous=False)
        X_discr = df_discr.drop(config.VAR_CLASSIF, axis=1)
        feat_names = list(X_discr)

        tic = perf_counter()
        sel_names = pymrmr.mRMR(df_discr, **sel_params)
        toc = perf_counter()

        time = toc - tic
        cols = np.array([name in sel_names for name in feat_names], dtype=float)

    elif feature_selector == 'filt-fcbf':
        df_discr, _ = discretize(df, leave_continuous=False)
        Y = df_discr[config.VAR_CLASSIF]
        X_discr = df_discr.drop(config.VAR_CLASSIF, axis=1)
        n = X_discr.shape[1]

        tic = perf_counter()
        idx, _ = fcbf(X_discr.to_numpy(), Y.to_numpy(), **sel_params)
        toc = perf_counter()

        time = toc - tic
        cols = np.zeros(n)
        cols[idx] = 1

    elif feature_selector == 'filt-relieff':  # core version of Relief-based algorithms
        try:
            df_discr, _ = discretize(df, leave_continuous=True)
        except ValueError:
            # discretize fails to handle NaNs, which exist if there was no imputation
            # but without imputation, discretization is not necessary
            df_discr = df.copy()
        Y = df_discr[config.VAR_CLASSIF]
        X_discr = df_discr.drop(config.VAR_CLASSIF, axis=1)
        n = X_discr.shape[1]

        selector = ReliefF(**sel_params)

        tic = perf_counter()
        selector.fit(X_discr.to_numpy(), Y.to_numpy())
        toc = perf_counter()

        time = toc - tic
        idx = selector.top_features_[:selector.n_features_to_select]
        cols = np.zeros(n)
        cols[idx] = 1

    elif feature_selector == 'filt-multisurf':  # multiclass extension to ReliefF, automatically determines the ideal number of neighbors
        try:
            df_discr, _ = discretize(df, leave_continuous=True)
        except ValueError:
            # discretize fails to handle NaNs, which exist if there was no imputation
            # but without imputation, discretization is not necessary
            df_discr = df.copy()
        Y = df_discr[config.VAR_CLASSIF]
        X_discr = df_discr.drop(config.VAR_CLASSIF, axis=1)
        n = X_discr.shape[1]

        selector = MultiSURF(**sel_params)

        tic = perf_counter()
        selector.fit(X_discr.to_numpy(), Y.to_numpy())
        toc = perf_counter()

        time = toc - tic
        idx = selector.top_features_[:selector.n_features_to_select]
        cols = np.zeros(n)
        cols[idx] = 1

    elif feature_selector == 'embed':
        sel_params['estimator'] = LogisticRegression(penalty='l1', solver='saga',
                                                     multi_class='multinomial', max_iter=10000,
                                                     C=selector_params['estimator'])
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
        if sel_params['estimator'] == 'LogReg':
            sel_params['estimator'] = FeatSelPipeline([('balancer', BALANCER_PIPE), ('estimator', LOGREG)])
        elif sel_params['estimator'] == 'KNNC':
            sel_params['estimator'] = FeatSelPipeline([('balancer', BALANCER_PIPE), ('estimator', KNNC)])
        elif sel_params['estimator'] == 'HGBC':
            sel_params['estimator'] = FeatSelPipeline([('balancer', BALANCER_PIPE), ('estimator', HGBC)])
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
        if sel_params['estimator'] == 'LogReg':
            sel_params['estimator'] = FeatSelPipeline([('balancer', BALANCER_PIPE), ('estimator', LOGREG)])
        elif sel_params['estimator'] == 'HGBC':
            sel_params['estimator'] = FeatSelPipeline([('balancer', BALANCER_PIPE), ('estimator', HGBC)])
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
        if sel_params['estimator'] == 'LogReg':
            sel_params['estimator'] = FeatSelPipeline([('balancer', BALANCER_PIPE), ('estimator', LOGREG)])
        elif sel_params['estimator'] == 'HGBC':
            sel_params['estimator'] = FeatSelPipeline([('balancer', BALANCER_PIPE), ('estimator', HGBC)])

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
        if sel_params['estimator'] == 'LogReg':
            sel_params['estimator'] = FeatSelPipeline([('balancer', BALANCER_PIPE), ('estimator', LOGREG)])
        elif sel_params['estimator'] == 'KNNC':
            sel_params['estimator'] = FeatSelPipeline([('balancer', BALANCER_PIPE), ('estimator', KNNC)])
        elif sel_params['estimator'] == 'HGBC':
            sel_params['estimator'] = FeatSelPipeline([('balancer', BALANCER_PIPE), ('estimator', HGBC)])
        else:
            raise NotImplementedError

        Y = df[config.VAR_CLASSIF]
        X = df.drop(config.VAR_CLASSIF, axis=1)

        tic = perf_counter()
        X_new, cols = bi.ga(X.to_numpy(), Y, **sel_params)
        toc = perf_counter()

        time = toc - tic

    elif feature_selector == 'wrap-bpso':
        if sel_params['estimator'] == 'LogReg':
            sel_params['estimator'] = FeatSelPipeline([('balancer', BALANCER_PIPE), ('estimator', LOGREG)])
        elif sel_params['estimator'] == 'HGBC':
            sel_params['estimator'] = FeatSelPipeline([('balancer', BALANCER_PIPE), ('estimator', HGBC)])
        elif sel_params['estimator'] == 'KNNC':
            sel_params['estimator'] = FeatSelPipeline([('balancer', BALANCER_PIPE), ('estimator', KNNC)])
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
