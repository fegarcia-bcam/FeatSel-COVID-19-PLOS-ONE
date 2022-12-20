import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import seaborn as sns

import config


IMG_FORMAT = 'tiff'
IMG_DPI = 300
IMG_EXTRA = {'compression': 'tiff_lzw'}

PATH = config.PATH_EXTRA_KNN
FILE = 'results_explore_knn.json'


if __name__ == '__main__':
    file_path = os.path.join(PATH, FILE)
    with open(file_path, 'r') as f_in:
        result = json.load(f_in)

    # read results
    df_knnc = pd.read_json(result['knnc'])
    df_knnr = pd.read_json(result['knnr'])

    # reformat
    df_knnc.insert(0, 'analysis', 'classif')
    df_knnr.insert(0, 'analysis', 'regress')
    df_knnc = df_knnc.reset_index(drop=False, names='iter')
    df_knnr = df_knnr.reset_index(drop=False, names='iter')
    df_knn = pd.concat([df_knnc, df_knnr]).sort_values(by=['iter', 'analysis'], ignore_index=True)

    knn_opts = np.sort(df_knn['k_opt'].unique())

    # hyperparams
    fig, ax = plt.subplots()
    sns.histplot(ax=ax, data=df_knn, x='k_opt', hue='analysis', multiple='dodge',
                 alpha=0.60, edgecolor='k', linewidth=1, palette=['green', 'red'])
    ax.set_xticks(knn_opts)
    ax.set_xlabel(r'Optimal $k$ in KNN')
    ax.set_ylabel('Occurrences')
    plt.show(block=True)

    path_fig = os.path.join(config.PATH_EXTRA_KNN, 'Histograms_KNN k_opt.{}'.format(IMG_FORMAT))
    fig.savefig(path_fig, format=IMG_FORMAT, dpi=IMG_DPI, pil_kwargs=IMG_EXTRA)

    # performance
    fig, ax = plt.subplots()
    box_plt = sns.boxplot(ax=ax, data=df_knn, x='k_opt', y='gms', hue='analysis',
                          boxprops={'alpha': 0.75}, palette=['green', 'red'])
    ax.set_xlabel(r'Optimal $k$ in KNN')
    ax.set_ylabel('GMS')
    plt.show(block=True)

    path_fig = os.path.join(config.PATH_EXTRA_KNN, 'Boxplots_KNN GMS.{}'.format(IMG_FORMAT))
    fig.savefig(path_fig, format=IMG_FORMAT, dpi=IMG_DPI, pil_kwargs=IMG_EXTRA)