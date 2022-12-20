import os
import json

import pandas as pd
import matplotlib.pyplot as plt

import analyse_results
import config


IMG_FORMAT = 'tiff'
IMG_DPI = 300
IMG_EXTRA = {'compression': 'tiff_lzw'}

PATH = config.PATH_RESULTS_WRA_GA

INDEX = [0.001, 0.020]

COLUMNS = ['Classification (LogReg) + knn', 'Classification (LogReg) + iterative',
           'Regression (Ridge) + knn', 'Regression (Ridge) + iterative',
           'Classification (KNNC) + knn', 'Classification (KNNC) + iterative',
           'Regression (KNNR) + knn', 'Regression (KNNR) + iterative',
           'HGB Classification (HGBC)',
           'HGB Regression (HGBR)']


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
        Pm = result['selector_params']['mutxpb']

        # count occurrences
        Z = result['sel_result']
        encoder = result['encoder']
        feat_occur = analyse_results.count_occurrences(Z=Z, feat_names=feat_names.values())
        result['feat_occur'] = feat_occur

        if estimator == 'LogReg':
            if imputer == 'knn':
                col = 'Classification (LogReg) + knn'
                label = 'GA Classif[LogReg;pm={:0.3f}] (knn) NFS no'.format(Pm)
            elif imputer == 'iterative':
                col = 'Classification (LogReg) + iterative'
                label = 'GA Classif[LogReg;pm={:0.3f}] (iter) NFS no'.format(Pm)
            else:
                raise RuntimeError

        elif estimator == 'Ridge':
            if imputer == 'knn':
                col = 'Regression (Ridge) + knn'
                label = 'GA Regress[Ridge;pm={:0.3f}] (knn) NFS no'.format(Pm)
            elif imputer == 'iterative':
                col = 'Regression (Ridge) + iterative'
                label = 'GA Regress[Ridge;pm={:0.3f}] (iter) NFS no'.format(Pm)
            else:
                raise RuntimeError

        elif estimator == 'KNNC':
            if imputer == 'knn':
                col = 'Classification (KNNC) + knn'
                label = 'GA Classif[KNN;pm={:0.3f}] (knn) NFS no'.format(Pm)
            elif imputer == 'iterative':
                col = 'Classification (KNNC) + iterative'
                label = 'GA Classif[KNN;pm={:0.3f}] (iter) NFS no'.format(Pm)
            else:
                raise RuntimeError

        elif estimator == 'KNNR':
            if imputer == 'knn':
                col = 'Regression (KNNR) + knn'
                label = 'GA Regress[KNN;pm={:0.3f}] (knn) NFS no'.format(Pm)
            elif imputer == 'iterative':
                col = 'Regression (KNNR) + iterative'
                label = 'GA Regress[KNN;pm={:0.3f}] (iter) NFS no'.format(Pm)
            else:
                raise RuntimeError

        elif estimator == 'HGBC':
            col = 'HGB Classification (HGBC)'
            label = 'GA Classif[HGB;pm={:0.3f}] NFS no'.format(Pm)

        elif estimator == 'HGBR':
            col = 'HGB Regression (HGBR)'
            label = 'GA Regress[HGB;pm={:0.3f}] NFS no'.format(Pm)

        else:
            raise RuntimeError

        df_stab_m[col][Pm] = stab_mean
        df_stab_ci[col][Pm] = stab_ci_d
        df_time_m[col][Pm] = t_mean
        df_time_ci[col][Pm] = t_ci_d

        file_out = config.FILE_RESULTS.format(label)
        file_path_out = os.path.join(PATH, file_out)
        with open(file_path_out, 'w') as f_out:
            json.dump(result, f_out)

    df_stab_m = df_stab_m.dropna(axis='columns', how='all')
    df_stab_ci = df_stab_ci.dropna(axis='columns', how='all')
    df_time_m = df_time_m.dropna(axis='columns', how='all')
    df_time_ci = df_time_ci.dropna(axis='columns', how='all')

    fig, ax = plt.subplots()
    df_stab_m.plot.bar(yerr=df_stab_ci, ax=ax, capsize=4, rot=0, colormap='RdYlGn')
    plt.ylabel('Stability')
    plt.xlabel(r'$P_{m}$')
    plt.title('Genetic algorithm')
    plt.show(block=True)

    path_fig = os.path.join(config.PATH_FIGURES, 'Stability_Wrapper [GA].{}'.format(IMG_FORMAT))
    fig.savefig(path_fig, format=IMG_FORMAT, dpi=IMG_DPI, pil_kwargs=IMG_EXTRA)

    fig, ax = plt.subplots()
    df_time_m.plot.bar(yerr=df_time_ci, ax=ax, capsize=4, rot=0, colormap='RdYlGn', logy=True)
    plt.ylabel('Computational time [s]')
    plt.xlabel(r'$P_{m}$')
    plt.title('Genetic algorithm')
    plt.show(block=True)

    path_fig = os.path.join(config.PATH_FIGURES, 'Time_Wrapper [GA].{}'.format(IMG_FORMAT))
    fig.savefig(path_fig, format=IMG_FORMAT, dpi=IMG_DPI, pil_kwargs=IMG_EXTRA)
