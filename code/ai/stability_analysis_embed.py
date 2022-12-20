import os
import json

import pandas as pd
import matplotlib.pyplot as plt

import analyse_results
import config


IMG_FORMAT = 'tiff'
IMG_DPI = 300
IMG_EXTRA = {'compression': 'tiff_lzw'}

PATH = config.PATH_RESULTS_EMB

INDEX_CLASSIF = [0.075, 0.050, 0.025, 0.010, 0.005]
INDEX_REGRESS = [0.005, 0.010, 0.025, 0.050, 0.075]

COLUMNS = ['knn', 'iterative']


if __name__ == '__main__':
    files = os.listdir(PATH)

    with open(config.FILE_FEAT_NAMES, 'r') as f:
        feat_names = json.load(f)

    df_stab_m_classif = pd.DataFrame(index=INDEX_CLASSIF, columns=COLUMNS)
    df_stab_ci_classif = pd.DataFrame(index=INDEX_CLASSIF, columns=COLUMNS)
    df_time_m_classif = pd.DataFrame(index=INDEX_CLASSIF, columns=COLUMNS)
    df_time_ci_classif = pd.DataFrame(index=INDEX_CLASSIF, columns=COLUMNS)

    df_stab_m_regress = pd.DataFrame(index=INDEX_REGRESS, columns=COLUMNS)
    df_stab_ci_regress = pd.DataFrame(index=INDEX_REGRESS, columns=COLUMNS)
    df_time_m_regress = pd.DataFrame(index=INDEX_REGRESS, columns=COLUMNS)
    df_time_ci_regress = pd.DataFrame(index=INDEX_REGRESS, columns=COLUMNS)

    for file_in in files:
        file_path_in = os.path.join(PATH, file_in)
        with open(file_path_in, 'r') as f_in:
            result = json.load(f_in)

        stab_mean, stab_var, stab_ci_d, sel_mean, sel_median, sel_std = analyse_results.get_stability_stats(result)
        t_mean, t_ci_d = analyse_results.get_time_stats(result)
        info = str(sel_mean) + r'$\pm$' + str(sel_std)

        analysis = result['analysis']
        imputer = result['imputer']
        val = result['selector_params']['estimator']

        # count occurrences
        Z = result['sel_result']
        encoder = result['encoder']
        feat_occur = analyse_results.count_occurrences(Z=Z, feat_names=feat_names.values())
        result['feat_occur'] = feat_occur

        if analysis == 'classif':
            if imputer == 'knn':
                label = 'Emb Classif[C={:0.4f}] (knn) NFS no'.format(val)
            elif imputer == 'iterative':
                label = 'Emb Classif[C={:0.4f}] (iter) NFS no'.format(val)
            else:
                raise RuntimeError
            df_stab_m_classif[imputer][val] = stab_mean
            df_stab_ci_classif[imputer][val] = stab_ci_d
            df_time_m_classif[imputer][val] = t_mean
            df_time_ci_classif[imputer][val] = t_ci_d

        elif analysis == 'regress':
            if imputer == 'knn':
                label = 'Emb Regress[a={:0.4f}] (knn) NFS no'.format(val)
            elif imputer == 'iterative':
                label = 'Emb Regress[a={:0.4f}] (iter) NFS no'.format(val)
            else:
                raise RuntimeError
            df_stab_m_regress[imputer][val] = stab_mean
            df_stab_ci_regress[imputer][val] = stab_ci_d
            df_time_m_regress[imputer][val] = t_mean
            df_time_ci_regress[imputer][val] = t_ci_d

        else:
            raise RuntimeError

        file_out = config.FILE_RESULTS.format(label)
        file_path_out = os.path.join(PATH, file_out)
        with open(file_path_out, 'w') as f_out:
            json.dump(result, f_out)

    fig, ax = plt.subplots()
    df_stab_m_classif.plot.bar(yerr=df_stab_ci_classif, ax=ax, capsize=4, rot=0, colormap='RdYlGn')
    plt.ylabel('Stability')
    plt.xlabel('C')
    plt.title('Classification (LogReg)')
    plt.show(block=True)

    path_fig = os.path.join(config.PATH_FIGURES, 'Stability_Embedded [LogReg].{}'.format(IMG_FORMAT))
    fig.savefig(path_fig, format=IMG_FORMAT, dpi=IMG_DPI, pil_kwargs=IMG_EXTRA)

    fig, ax = plt.subplots()
    df_stab_m_regress.plot.bar(yerr=df_stab_ci_regress, ax=ax, capsize=4, rot=0, colormap='RdYlGn')
    plt.ylabel('Stability')
    plt.xlabel(r'$\alpha$')
    plt.title('Regression (Lasso)')
    plt.show(block=True)

    path_fig = os.path.join(config.PATH_FIGURES, 'Stability_Embedded [Lasso].{}'.format(IMG_FORMAT))
    fig.savefig(path_fig, format=IMG_FORMAT, dpi=IMG_DPI, pil_kwargs=IMG_EXTRA)

    fig, ax = plt.subplots()
    df_time_m_classif.plot.bar(yerr=df_time_ci_classif, ax=ax, capsize=4, rot=0, colormap='RdYlGn', logy=True)
    plt.ylabel('Computational time [s]')
    plt.title('Classification (LogReg)')
    plt.xlabel('C')
    plt.show(block=True)

    path_fig = os.path.join(config.PATH_FIGURES, 'Time_Embedded [LogReg].{}'.format(IMG_FORMAT))
    fig.savefig(path_fig, format=IMG_FORMAT, dpi=IMG_DPI, pil_kwargs=IMG_EXTRA)

    fig, ax = plt.subplots()
    df_time_m_regress.plot.bar(yerr=df_time_ci_regress, ax=ax, capsize=4, rot=0, colormap='RdYlGn', logy=True)
    plt.ylabel('Computational time [s]')
    plt.xlabel(r'$\alpha$')
    plt.title('Regression (Lasso)')
    plt.show(block=True)

    path_fig = os.path.join(config.PATH_FIGURES, 'Time_Embedded [Lasso].{}'.format(IMG_FORMAT))
    fig.savefig(path_fig, format=IMG_FORMAT, dpi=IMG_DPI, pil_kwargs=IMG_EXTRA)
