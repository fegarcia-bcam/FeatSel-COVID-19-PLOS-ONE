import os
import shutil
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import analyse_results
import config


IMG_FORMAT = 'tiff'
IMG_DPI = 300
IMG_EXTRA = {'compression': 'tiff_lzw'}

PATH_BASE = config.PATH_RESULTS
PATH_EXCLUDE = [config.PATH_EXTRA, config.PATH_FIGURES, config.PATH_RESULTS_STABLE]


if __name__ == '__main__':
    # explore directory
    dirs_exclude = [os.path.basename(path_tmp) for path_tmp in PATH_EXCLUDE]
    files = []
    for folder, dirs, files_tmp in os.walk(PATH_BASE):  # loop recursively
        dirs[:] = [d for d in dirs if d not in dirs_exclude]  # exclude directory if it is in the exclude list
        for f_name in files_tmp:
            _, f_ext = os.path.splitext(f_name)
            if f_ext == '.json':  # check for extension
                file_path = os.path.join(folder, f_name)
                files.append(file_path)

    # filter only stable
    nfs_all = ['05', '10', '20', '40', 'no']
    results_stable = {nfs: [] for nfs in nfs_all}
    names_stable = {nfs: [] for nfs in nfs_all}

    for file_path in files:
        file_base = os.path.basename(file_path)
        info, _ = os.path.splitext(file_base)
        _, label = info.split('_')
        name, nfs = label.split(' NFS ')
        with open(file_path, 'r') as f:
            result = json.load(f)

        sel_result = result['sel_result']
        stabil_hypoth = analyse_results.test_higher(Z=sel_result, val_thr=config.STABIL_THRESH)
        if stabil_hypoth['reject']:
            results_stable[nfs].append(sel_result)
            names_stable[nfs].append(name)

            # copy to directory
            file_src = file_path
            file_dst = os.path.join(config.PATH_RESULTS_STABLE, file_base)
            shutil.copyfile(file_src, file_dst)

    # figures for different number of features
    for nfs, result_tmp in results_stable.items():
        n_sim = len(names_stable[nfs])
        df_sim = np.ones((n_sim, n_sim), dtype=float)
        df_sim = pd.DataFrame(data=df_sim, index=names_stable[nfs], columns=names_stable[nfs])
        for i in range(n_sim - 1):
            for j in range(i + 1, n_sim):
                Zi = result_tmp[i]
                Zj = result_tmp[j]
                _, jacc_mean, _, _ = analyse_results.get_similarities(Zi, Zj)
                df_sim.iloc[i, j] = jacc_mean
                df_sim.iloc[j, i] = jacc_mean

        fig_sim = sns.heatmap(df_sim, vmin=0.0, vmax=1.0, annot=True, square=True,
                              cmap='coolwarm', linewidths=0.5, annot_kws={'fontsize': 13})

        rot_x = 15 if nfs != 'no' else 10
        rot_y = 15 if nfs != 'no' else 80
        va_y = 'baseline' if nfs != 'no' else 'center'

        fig_sim.axes.set_xticklabels(labels=names_stable[nfs], rotation=rot_x, fontsize=8)
        fig_sim.axes.set_yticklabels(labels=names_stable[nfs], rotation=rot_y, fontsize=8, va=va_y)
        plt.show(block=True)

        file_fig = 'Similarity_NFS {}.{}'.format(nfs, IMG_FORMAT)
        path_fig = os.path.join(config.PATH_FIGURES, file_fig)
        fig_sim.figure.savefig(path_fig, format=IMG_FORMAT, dpi=IMG_DPI, pil_kwargs=IMG_EXTRA)
