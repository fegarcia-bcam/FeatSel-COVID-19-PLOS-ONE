import os
import json

import pandas as pd
import matplotlib.pyplot as plt

import analyse_results
import config


IMG_FORMAT = 'tiff'
IMG_DPI = 300
IMG_EXTRA = {'compression': 'tiff_lzw'}

PATH = config.PATH_RESULTS_FIL

INDEX_A = [5, 10, 20, 40, 'non-determ']
INDEX_B = [5, 10, 20, 40]

COLUMNS_A = ['MI Classif', 'MI Regress', 'mRMR-MID', 'mRMR-MIQ', 'FCBF']
COLUMNS_B = ['ReliefF10', 'ReliefF100', 'MultiSURF']


if __name__ == '__main__':
    files = os.listdir(PATH)

    with open(config.FILE_FEAT_NAMES, 'r') as f:
        feat_names = json.load(f)

    df_stab_m_knn = pd.DataFrame(index=INDEX_A, columns=COLUMNS_A)
    df_stab_ci_knn = pd.DataFrame(index=INDEX_A, columns=COLUMNS_A)
    df_time_m_knn = pd.DataFrame(index=INDEX_A, columns=COLUMNS_A)
    df_time_ci_knn = pd.DataFrame(index=INDEX_A, columns=COLUMNS_A)

    df_stab_m_iter = pd.DataFrame(index=INDEX_A, columns=COLUMNS_A)
    df_stab_ci_iter = pd.DataFrame(index=INDEX_A, columns=COLUMNS_A)
    df_time_m_iter = pd.DataFrame(index=INDEX_A, columns=COLUMNS_A)
    df_time_ci_iter = pd.DataFrame(index=INDEX_A, columns=COLUMNS_A)

    df_stab_m_none = pd.DataFrame(index=INDEX_B, columns=COLUMNS_B)
    df_stab_ci_none = pd.DataFrame(index=INDEX_B, columns=COLUMNS_B)
    df_time_m_none = pd.DataFrame(index=INDEX_B, columns=COLUMNS_B)
    df_time_ci_none = pd.DataFrame(index=INDEX_B, columns=COLUMNS_B)

    for file_in in files:
        file_path_in = os.path.join(PATH, file_in)
        with open(file_path_in, 'r') as f_in:
            result = json.load(f_in)

        stab_mean, stab_var, stab_ci_d, sel_mean, sel_median, sel_std = analyse_results.get_stability_stats(result)
        t_mean, t_ci_d = analyse_results.get_time_stats(result)

        imputer = result['imputer']
        feature_selector = result['feature_selector']

        # count occurrences
        Z = result['sel_result']
        encoder = result['encoder']
        feat_occur = analyse_results.count_occurrences(Z=Z, feat_names=feat_names.values())
        result['feat_occur'] = feat_occur

        if imputer == 'knn':
            if feature_selector == 'filter-univar':
                n_feat = result['selector_params']['param']
                a = result['analysis']
                if a == 'classif':
                    col = 'MI Classif'
                    label = 'MI Classif (knn) NFS {:02d}'.format(n_feat)
                elif a == 'regress':
                    col = 'MI Regress'
                    label = 'MI Regress (knn) NFS {:02d}'.format(n_feat)
                else:
                    raise RuntimeError

            elif feature_selector == 'filt-mrmr':
                n_feat = result['selector_params']['nfeats']
                a = result['selector_params']['method']
                if a == 'MID':
                    col = 'mRMR-MID'
                    label = 'mRMR[MID] (knn) NFS {:02d}'.format(n_feat)
                elif a == 'MIQ':
                    col = 'mRMR-MIQ'
                    label = 'mRMR[MIQ] (knn) NFS {:02d}'.format(n_feat)
                else:
                    raise RuntimeError

            elif feature_selector == 'filt-fcbf':
                n_feat = 'non-determ'
                col = 'FCBF'
                label = 'FCBF (knn) NFS no'

            else:
                raise RuntimeError

            df_stab_m_knn[col][n_feat] = stab_mean
            df_stab_ci_knn[col][n_feat] = stab_ci_d
            df_time_m_knn[col][n_feat] = t_mean
            df_time_ci_knn[col][n_feat] = t_ci_d

        elif imputer == 'iterative':
            if feature_selector == 'filter-univar':
                n_feat = result['selector_params']['param']
                a = result['analysis']
                if a == 'classif':
                    col = 'MI Classif'
                    label = 'MI Classif (iter) NFS {:02d}'.format(n_feat)
                elif a == 'regress':
                    col = 'MI Regress'
                    label = 'MI Regress (iter) NFS {:02d}'.format(n_feat)

                else:
                    raise RuntimeError

            elif feature_selector == 'filt-mrmr':
                n_feat = result['selector_params']['nfeats']
                a = result['selector_params']['method']
                if a == 'MID':
                    col = 'mRMR-MID'
                    label = 'mRMR[MID] (iter) NFS {:02d}'.format(n_feat)
                elif a == 'MIQ':
                    col = 'mRMR-MIQ'
                    label = 'mRMR[MIQ] (iter) NFS {:02d}'.format(n_feat)
                else:
                    raise RuntimeError

            elif feature_selector == 'filt-fcbf':
                n_feat = 'non-determ'
                col = 'FCBF'
                label = 'FCBF (iter) NFS no'

            else:
                raise RuntimeError

            df_stab_m_iter[col][n_feat] = stab_mean
            df_stab_ci_iter[col][n_feat] = stab_ci_d
            df_time_m_iter[col][n_feat] = t_mean
            df_time_ci_iter[col][n_feat] = t_ci_d

        elif imputer == 'none':
            n_feat = result['selector_params']['n_features_to_select']
            if feature_selector == 'filt-relieff':
                a = result['selector_params']['n_neighbors']
                if a == 10:
                    col = 'ReliefF10'
                    label = 'ReliefF[k=10] NFS {:02d}'.format(n_feat)
                elif a == 100:
                    col = 'ReliefF100'
                    label = 'ReliefF[k=100] NFS {:02d}'.format(n_feat)
                else:
                    raise RuntimeError

            elif feature_selector == 'filt-multisurf':
                col = 'MultiSURF'
                label = 'MultiSURF NFS {:02d}'.format(n_feat)

            else:
                raise RuntimeError

            df_stab_m_none[col][n_feat] = stab_mean
            df_stab_ci_none[col][n_feat] = stab_ci_d
            df_time_m_none[col][n_feat] = t_mean
            df_time_ci_none[col][n_feat] = t_ci_d

        else:
            raise RuntimeError

        file_out = config.FILE_RESULTS.format(label)
        file_path_out = os.path.join(PATH, file_out)
        with open(file_path_out, 'w') as f_out:
            json.dump(result, f_out)

    fig, ax = plt.subplots()
    df_stab_m_knn.plot.bar(yerr=df_stab_ci_knn, ax=ax, capsize=4, rot=0, colormap='RdYlGn')
    plt.ylabel('Stability')
    plt.xlabel('Number of features')
    plt.title('Filters: knn imputer')
    plt.show(block=True)

    path_fig = os.path.join(config.PATH_FIGURES, 'Stability_Filters [MI, mRMR, FCBF] (knn).{}'.format(IMG_FORMAT))
    fig.savefig(path_fig, format=IMG_FORMAT, dpi=IMG_DPI, pil_kwargs=IMG_EXTRA)

    fig, ax = plt.subplots()
    df_stab_m_iter.plot.bar(yerr=df_stab_ci_iter, ax=ax, capsize=4, rot=0, colormap='RdYlGn')
    plt.ylabel('Stability')
    plt.xlabel('Number of features')
    plt.title('Filters: iterative imputer')
    plt.show(block=True)

    path_fig = os.path.join(config.PATH_FIGURES, 'Stability_Filters [MI, mRMR, FCBF] (iter).{}'.format(IMG_FORMAT))
    fig.savefig(path_fig, format=IMG_FORMAT, dpi=IMG_DPI, pil_kwargs=IMG_EXTRA)

    fig, ax = plt.subplots()
    df_stab_m_none.plot.bar(yerr=df_stab_ci_none, ax=ax, capsize=4, rot=0, colormap='RdYlGn')
    plt.ylabel('Stability')
    plt.xlabel('Number of features')
    plt.title('Filters: no imputation')
    plt.show(block=True)

    path_fig = os.path.join(config.PATH_FIGURES, 'Stability_Filters [RBA].{}'.format(IMG_FORMAT))
    fig.savefig(path_fig, format=IMG_FORMAT, dpi=IMG_DPI, pil_kwargs=IMG_EXTRA)

    fig, ax = plt.subplots()
    df_time_m_knn.plot.bar(yerr=df_time_ci_knn, ax=ax, capsize=4, rot=0, colormap='RdYlGn', logy=True)
    plt.ylabel('Computational time [s]')
    plt.xlabel('Number of features')
    plt.title('Filters: knn imputer')
    plt.show(block=True)

    path_fig = os.path.join(config.PATH_FIGURES, 'Time_Filters [MI, mRMR, FCBF] (knn).{}'.format(IMG_FORMAT))
    fig.savefig(path_fig, format=IMG_FORMAT, dpi=IMG_DPI, pil_kwargs=IMG_EXTRA)

    fig, ax = plt.subplots()
    df_time_m_iter.plot.bar(yerr=df_time_ci_iter, ax=ax, capsize=4, rot=0, colormap='RdYlGn', logy=True)
    plt.ylabel('Computational time [s]')
    plt.xlabel('Number of features')
    plt.title('Filters: iterative imputer')
    plt.show(block=True)

    path_fig = os.path.join(config.PATH_FIGURES, 'Time_Filters [MI, mRMR, FCBF] (iter).{}'.format(IMG_FORMAT))
    fig.savefig(path_fig, format=IMG_FORMAT, dpi=IMG_DPI, pil_kwargs=IMG_EXTRA)

    fig, ax = plt.subplots()
    df_time_m_none.plot.bar(yerr=df_time_ci_none, ax=ax, capsize=4, rot=0, colormap='RdYlGn', logy=True)
    plt.ylabel('Computational time [s]')
    plt.xlabel('Number of features')
    plt.title('Filters: no imputation')
    plt.show(block=True)

    path_fig = os.path.join(config.PATH_FIGURES, 'Time_Filters [RBA].{}'.format(IMG_FORMAT))
    fig.savefig(path_fig, format=IMG_FORMAT, dpi=IMG_DPI, pil_kwargs=IMG_EXTRA)
