import os
import json

import pandas as pd
import matplotlib.pyplot as plt

import analyse_results
import config


IMG_FORMAT = 'tiff'
IMG_DPI = 300
IMG_EXTRA = {'compression': 'tiff_lzw'}

PATH = config.PATH_RESULTS_WRA_BPSO

INDEX = [2, 6]

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
        Vm = result['selector_params']['velocity_clamp'][1]

        # count occurrences
        Z = result['sel_result']
        encoder = result['encoder']
        feat_occur = analyse_results.count_occurrences(Z=Z, feat_names=feat_names.values())
        result['feat_occur'] = feat_occur

        if estimator == 'LogReg':
            if imputer == 'knn':
                col = 'Classification (LogReg) + knn'
                label = 'BPSO Classif[LogReg;Vm={}] (knn) NFS no'.format(Vm)
            elif imputer == 'iterative':
                col = 'Classification (LogReg) + iterative'
                label = 'BPSO Classif[LogReg;Vm={}] (iter) NFS no'.format(Vm)
            else:
                raise RuntimeError

        elif estimator == 'Ridge':
            if imputer == 'knn':
                col = 'Regression (Ridge) + knn'
                label = 'BPSO Regress[Ridge;Vm={}] (knn) NFS no'.format(Vm)
            elif imputer == 'iterative':
                col = 'Regression (Ridge) + iterative'
                label = 'BPSO Regress[Ridge;Vm={}] (iter) NFS no'.format(Vm)
            else:
                raise RuntimeError

        elif estimator == 'KNNC':
            if imputer == 'knn':
                col = 'Classification (KNNC) + knn'
                label = 'BPSO Classif[KNN;Vm={}] (knn) NFS no'.format(Vm)
            elif imputer == 'iterative':
                col = 'Classification (KNNC) + iterative'
                label = 'BPSO Classif[KNN;Vm={}] (iter) NFS no'.format(Vm)
            else:
                raise RuntimeError

        elif estimator == 'KNNR':
            if imputer == 'knn':
                col = 'Regression (KNNR) + knn'
                label = 'BPSO Regress[KNN;Vm={}] (knn) NFS no'.format(Vm)
            elif imputer == 'iterative':
                col = 'Regression (KNNR) + iterative'
                label = 'BPSO Regress[KNN;Vm={}] (iter) NFS no'.format(Vm)
            else:
                raise RuntimeError

        elif estimator == 'HGBC':
            col = 'HGB Classification (HGBC)'
            label = 'BPSO Classif[HGB;Vm={}] NFS no'.format(Vm)

        elif estimator == 'HGBR':
            col = 'HGB Regression (HGBR)'
            label = 'BPSO Regress[HGB;Vm={}] NFS no'.format(Vm)

        else:
            raise RuntimeError

        df_stab_m[col][Vm] = stab_mean
        df_stab_ci[col][Vm] = stab_ci_d
        df_time_m[col][Vm] = t_mean
        df_time_ci[col][Vm] = t_ci_d

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
    plt.xlabel(r'$|v|_{max}$')
    plt.title('Particle swarm optimization')
    plt.show(block=True)

    path_fig = os.path.join(config.PATH_FIGURES, 'Stability_Wrapper [BPSO].{}'.format(IMG_FORMAT))
    fig.savefig(path_fig, format=IMG_FORMAT, dpi=IMG_DPI, pil_kwargs=IMG_EXTRA)

    fig, ax = plt.subplots()
    df_time_m.plot.bar(yerr=df_time_ci, ax=ax, capsize=4, rot=0, colormap='RdYlGn', logy=True)
    plt.ylabel('Computational time [s]')
    plt.xlabel(r'$|v|_{max}$')
    plt.title('Particle swarm optimization')
    plt.show(block=True)

    path_fig = os.path.join(config.PATH_FIGURES, 'Time_Wrapper [BPSO].{}'.format(IMG_FORMAT))
    fig.savefig(path_fig, format=IMG_FORMAT, dpi=IMG_DPI, pil_kwargs=IMG_EXTRA)
