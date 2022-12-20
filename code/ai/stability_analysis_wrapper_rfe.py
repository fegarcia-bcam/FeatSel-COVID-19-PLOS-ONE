import os
import json

import pandas as pd
import matplotlib.pyplot as plt

import analyse_results
import config


IMG_FORMAT = 'tiff'
IMG_DPI = 300
IMG_EXTRA = {'compression': 'tiff_lzw'}

PATH = config.PATH_RESULTS_WRA_RFE

INDEX = [5, 10, 20, 40]

COLUMNS = ['Classification (LogReg) + knn', 'Classification (LogReg) + iterative',
           'Regression (Ridge) + knn', 'Regression (Ridge) + iterative']


if __name__ == '__main__':
    files = os.listdir(PATH)

    with open(config.FILE_FEAT_NAMES, 'r') as f:
        feat_names = json.load(f)

    df_stab_m = pd.DataFrame(index=INDEX, columns=COLUMNS)
    df_stab_ci = pd.DataFrame(index=INDEX, columns=COLUMNS)
    df_time_m = pd.DataFrame(index=INDEX, columns=COLUMNS)
    df_time_ci = pd.DataFrame(index=INDEX, columns=COLUMNS)

    for file_in in files:
        file_path_in = os.path.join(PATH, file_in)
        with open(file_path_in, 'r') as f_in:
            result = json.load(f_in)

        stab_mean, stab_var, stab_ci_d, sel_mean, sel_median, sel_std = analyse_results.get_stability_stats(result)
        t_mean, t_ci_d = analyse_results.get_time_stats(result)

        imputer = result['imputer']
        estimator = result['selector_params']['estimator']
        n_feat = result['selector_params']['n_features_to_select']

        # count occurrences
        Z = result['sel_result']
        encoder = result['encoder']
        feat_occur = analyse_results.count_occurrences(Z=Z, feat_names=feat_names.values())
        result['feat_occur'] = feat_occur

        if estimator == 'LogReg':
            if imputer == 'knn':
                col = 'Classification (LogReg) + knn'
                label = 'RFE Classif (knn) NFS {:02d}'.format(n_feat)
            elif imputer == 'iterative':
                col = 'Classification (LogReg) + iterative'
                label = 'RFE Classif (iter) NFS {:02d}'.format(n_feat)
            else:
                raise RuntimeError

        elif estimator == 'Ridge':
            if imputer == 'knn':
                col = 'Regression (Ridge) + knn'
                label = 'RFE Regress (knn) NFS {:02d}'.format(n_feat)
            elif imputer == 'iterative':
                col = 'Regression (Ridge) + iterative'
                label = 'RFE Regress (iter) NFS {:02d}'.format(n_feat)
            else:
                raise RuntimeError

        else:
            raise RuntimeError

        df_stab_m[col][n_feat] = stab_mean
        df_stab_ci[col][n_feat] = stab_ci_d
        df_time_m[col][n_feat] = t_mean
        df_time_ci[col][n_feat] = t_ci_d

        file_out = config.FILE_RESULTS.format(label)
        file_path_out = os.path.join(PATH, file_out)
        with open(file_path_out, 'w') as f_out:
            json.dump(result, f_out)

    fig, ax = plt.subplots()
    df_stab_m.plot.bar(yerr=df_stab_ci, ax=ax, capsize=4, rot=0, colormap='RdYlGn')
    plt.ylabel('Stability')
    plt.xlabel('Number of features')
    plt.title('Recursive feature elimination')
    plt.show(block=True)

    path_fig = os.path.join(config.PATH_FIGURES, 'Stability_Wrapper [RFE].{}'.format(IMG_FORMAT))
    fig.savefig(path_fig, format=IMG_FORMAT, dpi=IMG_DPI, pil_kwargs=IMG_EXTRA)

    fig, ax = plt.subplots()
    df_time_m.plot.bar(yerr=df_time_ci, ax=ax, capsize=4, rot=0, colormap='RdYlGn', logy=True)
    plt.ylabel('Computational time [s]')
    plt.xlabel('Number of features')
    plt.title('Recursive feature elimination')
    plt.show(block=True)

    path_fig = os.path.join(config.PATH_FIGURES, 'Time_Wrapper [RFE].{}'.format(IMG_FORMAT))
    fig.savefig(path_fig, format=IMG_FORMAT, dpi=IMG_DPI, pil_kwargs=IMG_EXTRA)
