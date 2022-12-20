import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config


IMG_FORMAT = 'tiff'
IMG_DPI = 300
IMG_EXTRA = {'compression': 'tiff_lzw'}

PATH = config.PATH_EXTRA_LINEAR
FILE = 'results_explore_linear.json'


if __name__ == '__main__':
    file_path = os.path.join(PATH, FILE)
    with open(file_path, 'r') as f_in:
        result = json.load(f_in)

    # read results
    df_c = pd.read_json(result['c'])
    df_a = pd.read_json(result['a'])

    # reformat
    df_c.columns = pd.MultiIndex.from_tuples([('grid', 'c'), ('grid', 'gms'), ('bayes', 'c'), ('bayes', 'gms')])
    df_a.columns = pd.MultiIndex.from_tuples([('grid', 'a'), ('grid', 'gms'), ('bayes', 'a'), ('bayes', 'gms')])
    df_c = df_c.stack(level=-2)
    df_a = df_a.stack(level=-2)
    df_c.index.rename(['iter', 'search'], inplace=True)
    df_a.index.rename(['iter', 'search'], inplace=True)

    # classification, hyperparams and agreement
    c_grid = df_c.loc[(slice(None), 'grid'), 'c'].to_numpy()
    c_bayes = df_c.loc[(slice(None), 'bayes'), 'c'].to_numpy()

    _, bins_log_c_gr = np.histogram(np.log10(c_grid), bins='doane')
    _, bins_log_c_by = np.histogram(np.log10(c_bayes), bins='doane')
    bins_nat_c_gr = 10.0 ** bins_log_c_gr
    bins_nat_c_by = 10.0 ** bins_log_c_by
    counts_nat_c_gr, _ = np.histogram(c_grid, bins=bins_nat_c_gr)
    counts_nat_c_by, _ = np.histogram(c_bayes, bins=bins_nat_c_by)

    fig, ax = plt.subplots()
    ax.hist(c_grid, bins=bins_nat_c_gr, label='Grid', alpha=0.30, edgecolor='k', linewidth=1)
    ax.hist(c_bayes, bins=bins_nat_c_by, label='Bayes', alpha=0.30, edgecolor='k', linewidth=1)
    ax.set_xscale('log')
    ax.set_xlabel('Optimal C')
    ax.set_ylabel('Occurrences')
    ax.legend()
    plt.show(block=True)

    path_fig = os.path.join(config.PATH_EXTRA_LINEAR, 'Histograms_Classif C.{}'.format(IMG_FORMAT))
    fig.savefig(path_fig, format=IMG_FORMAT, dpi=IMG_DPI, pil_kwargs=IMG_EXTRA)

    fig, ax = plt.subplots()
    ax.axline((0, 0), (1, 1), color='k', linestyle=':')
    ax.scatter(c_grid, c_bayes, color='g', marker='.')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Optimal C: Grid')
    ax.set_ylabel('Optimal C: Bayes')
    ax.set_aspect('equal')
    plt.grid()
    plt.show(block=True)

    path_fig = os.path.join(config.PATH_EXTRA_LINEAR, 'Scatter_Classif C.{}'.format(IMG_FORMAT))
    fig.savefig(path_fig, format=IMG_FORMAT, dpi=IMG_DPI, pil_kwargs=IMG_EXTRA)

    # regression, hyperparams and agreement
    a_grid = df_a.loc[(slice(None), 'grid'), 'a'].to_numpy()
    a_bayes = df_a.loc[(slice(None), 'bayes'), 'a'].to_numpy()

    _, bins_log_a_gr = np.histogram(np.log10(a_grid), bins='doane')
    _, bins_log_a_by = np.histogram(np.log10(a_bayes), bins='doane')
    bins_nat_a_gr = 10.0 ** bins_log_a_gr
    bins_nat_a_by = 10.0 ** bins_log_a_by
    counts_nat_a_gr, _ = np.histogram(a_grid, bins=bins_nat_a_gr)
    counts_nat_a_by, _ = np.histogram(a_bayes, bins=bins_nat_a_by)

    fig, ax = plt.subplots()
    ax.hist(a_grid, bins=bins_nat_a_gr, label='Grid', alpha=0.30, edgecolor='k', linewidth=1)
    ax.hist(a_bayes, bins=bins_nat_a_by, label='Bayes', alpha=0.30, edgecolor='k', linewidth=1)
    ax.set_xscale('log')
    ax.set_xlabel(r'Optimal $\alpha$')
    ax.legend()
    plt.show(block=True)

    path_fig = os.path.join(config.PATH_EXTRA_LINEAR, 'Histograms_Regress Alpha.{}'.format(IMG_FORMAT))
    fig.savefig(path_fig, format=IMG_FORMAT, dpi=IMG_DPI, pil_kwargs=IMG_EXTRA)

    fig, ax = plt.subplots()
    ax.axline((0, 0), (1, 1), color='k', linestyle=':')
    ax.scatter(a_grid, a_bayes, color='g', marker='.')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Optimal $\alpha$: Grid')
    ax.set_ylabel(r'Optimal $\alpha$: Bayes')
    ax.set_aspect('equal')
    plt.grid()
    plt.show(block=True)

    path_fig = os.path.join(config.PATH_EXTRA_LINEAR, 'Scatter_Regress Alpha.{}'.format(IMG_FORMAT))
    fig.savefig(path_fig, format=IMG_FORMAT, dpi=IMG_DPI, pil_kwargs=IMG_EXTRA)

    # classification, performance
    score_c_grid = df_c.loc[(slice(None), 'grid'), 'gms'].to_numpy()
    score_c_bayes = df_c.loc[(slice(None), 'bayes'), 'gms'].to_numpy()

    fig, ax = plt.subplots()
    ax.hist(score_c_grid, bins='doane', label='Grid', alpha=0.30, edgecolor='k', linewidth=1)
    ax.hist(score_c_bayes, bins='doane', label='Bayes', alpha=0.30, edgecolor='k', linewidth=1)
    ax.set_xlabel(r'GMS score')
    ax.legend()
    plt.show(block=True)

    path_fig = os.path.join(config.PATH_EXTRA_LINEAR, 'Histograms_Classif GMS.{}'.format(IMG_FORMAT))
    fig.savefig(path_fig, format=IMG_FORMAT, dpi=IMG_DPI, pil_kwargs=IMG_EXTRA)

    fig, ax = plt.subplots()
    ax.scatter(c_grid, score_c_grid, color='tab:blue', marker='.', label='Grid')
    ax.scatter(c_bayes, score_c_bayes, color='tab:orange', marker='.', label='Bayes')
    ax.set_xscale('log')
    ax.set_xlabel('Optimal C')
    ax.set_ylabel('GMS score')
    ax.legend()
    plt.grid()
    plt.show(block=True)

    path_fig = os.path.join(config.PATH_EXTRA_LINEAR, 'Scatter_Classif GMS.{}'.format(IMG_FORMAT))
    fig.savefig(path_fig, format=IMG_FORMAT, dpi=IMG_DPI, pil_kwargs=IMG_EXTRA)

    # regression, performance
    score_a_grid = df_a.loc[(slice(None), 'grid'), 'gms'].to_numpy()
    score_a_bayes = df_a.loc[(slice(None), 'bayes'), 'gms'].to_numpy()

    fig, ax = plt.subplots()
    ax.hist(score_a_grid, bins='doane', label='Grid', alpha=0.30, edgecolor='k', linewidth=1)
    ax.hist(score_a_bayes, bins='doane', label='Bayes', alpha=0.30, edgecolor='k', linewidth=1)
    ax.set_xlabel(r'GMS score')
    ax.legend()
    plt.show(block=True)

    path_fig = os.path.join(config.PATH_EXTRA_LINEAR, 'Histograms_Regress GMS.{}'.format(IMG_FORMAT))
    fig.savefig(path_fig, format=IMG_FORMAT, dpi=IMG_DPI, pil_kwargs=IMG_EXTRA)

    fig, ax = plt.subplots()
    ax.scatter(a_grid, score_a_grid, color='tab:blue', marker='.', label='Grid')
    ax.scatter(a_bayes, score_a_bayes, color='tab:orange', marker='.', label='Bayes')
    ax.set_xscale('log')
    ax.set_xlabel(r'Optimal $\alpha$')
    ax.set_ylabel('GMS score')
    ax.legend()
    plt.grid()
    plt.show(block=True)

    path_fig = os.path.join(config.PATH_EXTRA_LINEAR, 'Scatter_Regress GMS.{}'.format(IMG_FORMAT))
    fig.savefig(path_fig, format=IMG_FORMAT, dpi=IMG_DPI, pil_kwargs=IMG_EXTRA)
