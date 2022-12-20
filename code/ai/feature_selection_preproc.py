import numpy as np

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

from imblearn.over_sampling import RandomOverSampler

import config


def preprocess(X_in, y_in, encoder, scaler, imputer, imputer_params, balancer, balancer_params):
    X_out = X_in.copy()

    # column transformation by one-hot encoding for categorical variables
    if encoder == 'one-hot':
        encoder_obj = config.ENCODER_ONEHOT
    else:
        raise NotImplementedError

    # apply column encoding
    X_out = encoder_obj.fit_transform(X_out)

    # scaler
    if scaler == 'none':
        scaler_obj = None
    elif scaler == 'standard':
        scaler_obj = StandardScaler()
    elif scaler == 'robust':
        scaler_obj = RobustScaler()
    else:
        raise NotImplementedError

    # apply scaling
    if scaler_obj is not None:
        X_out = scaler_obj.fit_transform(X_out)

    # imputer
    if imputer == 'none':
        imputer_obj = None
    elif imputer == 'simple':
        imputer_obj = SimpleImputer(**imputer_params)
    elif imputer == 'knn':
        imputer_obj = KNNImputer(**imputer_params)
    elif imputer == 'iterative':
        imputer_obj = IterativeImputer(**imputer_params)
    else:
        raise NotImplementedError

    # apply imputation
    feat_has_nans = np.isnan(X_out).any(axis=0).tolist()
    num_pats = X_out.shape[0]

    # find all NaN columns
    is_nan_feat = np.all(np.isnan(X_out), axis=0)
    idx_nan_feat = np.argwhere(is_nan_feat)

    if imputer_obj is not None:
        X_out = imputer_obj.fit_transform(X_out)

    # fill with zeros
    col_zeros = np.zeros((num_pats, 1))
    for idx in idx_nan_feat:
        X_out = np.hstack((X_out[:, :idx[0]], col_zeros, X_out[:, idx[0]:]))

    # get feature names
    if encoder == 'one-hot':
        feat_names_in = encoder_obj.get_feature_names_out()
    else:
        raise NotImplementedError

    # include names for those extra vars newly created by the missing indicator
    feat_names_out = feat_names_in.copy()
    if (imputer_obj is not None) and ('add_indicator' in imputer_params) and (imputer_params['add_indicator']):
        for f_name_in, f_has_nan in zip(feat_names_in, feat_has_nans):
            if f_has_nan:
                f_name_out = f_name_in + '_miss'
                feat_names_out.append(f_name_out)

    # final name formatting
    feat_names_out = [f_name_out.replace('_x0_', '').replace(' ', '-').lower() for f_name_out in feat_names_out]

    # balancer
    if balancer == 'none':
        balancer_obj = None
    elif balancer == 'rand-oversample':
        balancer_obj = RandomOverSampler(**balancer_params)
    else:
        raise NotImplementedError

    # apply balance
    y_out = y_in.copy()
    if balancer_obj is not None:
        X_out, y_out = balancer_obj.fit_resample(X_out, y_in)

    return X_out, y_out, feat_names_out
