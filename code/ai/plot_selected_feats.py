import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

import analyse_results
import config


IMG_FORMAT = 'tiff'
IMG_DPI = 300
IMG_EXTRA = {'compression': 'tiff_lzw'}

FIGURE_SIZE = (6, 8)

PATH = config.PATH_RESULTS_STABLE


if __name__ == '__main__':
    results_stable = []
    names_stable = []

    files = os.listdir(PATH)
    for file in files:
        file_path = os.path.join(PATH, file)
        info, _ = os.path.splitext(file)
        _, name = info.split('_')
        with open(file_path, 'r') as f:
            result = json.load(f)

        results_stable.append({'n_iters': len(result['sel_result']), 'feat_occur': result['feat_occur']})
        names_stable.append(name)

    with open(config.FILE_FEAT_NAMES, 'r') as f:
        feat_names = json.load(f)
    num_feats = len(feat_names)

    algorithms = {'filt-univ': ['MI'],
                  'filt-multi': ['FCBF', 'mRMR[MID]', 'mRMR[MIQ]', 'ReliefF[k=10]', 'ReliefF[k=100]', 'MultiSURF'],
                  'embed': ['Emb'],
                  'wrap': ['SFS', 'RFE', 'RFECV', 'GA', 'BPSO']}
    df_feat_count = pd.DataFrame(data=0, index=range(num_feats), columns=algorithms.keys())
    s_algor_count = pd.Series(data=0, index=algorithms.keys())

    for result, name in zip(results_stable, names_stable):
        feat_occur = result['feat_occur']
        n_iters = result['n_iters']
        if n_iters != config.N_BOOTSTR:
            raise RuntimeError  # should never happen! (but we check, nevertheless)

        thr = config.FREQ_FEATS_THRESH * n_iters
        filt_occur = analyse_results.filter_by_occur(feat_occur, thr=thr)

        algorithm_name, _ = name.split(' ', maxsplit=1)
        algorithm_type = None
        for algor_key, algor_gr in algorithms.items():
            if algorithm_name in algor_gr:
                algorithm_type = algor_key
                break
        if algorithm_type is None:
            raise RuntimeError  # should never happen! (but we check, nevertheless)
        s_algor_count[algorithm_type] += 1

        # store results
        for idx_feat in filt_occur.keys():
            df_feat_count.loc[idx_feat, algorithm_type] += 1

        # process data
        df = pd.DataFrame.from_dict({'occur': filt_occur})
        df /= n_iters
        df *= 100.0
        df = df.rename(columns={'occur': 'freq'}).reset_index(drop=False)
        df = df.sort_values(by=['freq', 'index'], ascending=[False, True])
        num_feats_sel = len(df.index)

        # plot
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        ax.set_xlabel('Frequency of selection (%)', fontsize=13)
        ax.set_ylabel('Feature number', fontsize=13)

        x_bar = np.arange(num_feats_sel)
        y_bar = np.flip(df['freq'].to_numpy())

        c_map = cm.get_cmap('coolwarm')
        c_bar = [c_map(y / 100.0) for y in y_bar]
        ax.barh(x_bar, y_bar, color=c_bar, alpha=0.75)

        x_delta = 5.0
        x_max = 100.0
        x_min = x_delta * np.floor(100.0 * config.FREQ_FEATS_THRESH / x_delta)
        num_x_ticks = (x_max - x_min) / x_delta + 1
        x_ticks_bar = np.linspace(x_min, x_max, int(num_x_ticks))

        y_ticks_labels = np.flip(df['index'].to_numpy())

        ax.set_xlim(xmin=x_min, xmax=x_max)
        ax.set_xticks(x_ticks_bar, fontsize=13, rotation=0)
        ax.set_yticks(x_bar, labels=y_ticks_labels, fontsize=13, rotation=15)

        plt.grid(axis='x', alpha=0.60)
        plt.show(block=True)

        file_fig = 'FreqFeats_{}.{}'.format(name, IMG_FORMAT)
        path_fig = os.path.join(config.PATH_FIGURES, file_fig)
        fig.savefig(path_fig, format=IMG_FORMAT, dpi=IMG_DPI, pil_kwargs=IMG_EXTRA)

    # rank features by selected ratio
    num_algor = s_algor_count.sum()
    df_feat_count['total'] = df_feat_count.sum(axis='columns')
    df_feat_count['ratio'] = 100.0 * df_feat_count['total'] / num_algor
    df_feat_count = df_feat_count.reset_index(drop=False)
    df_feat_count = df_feat_count.sort_values(by=['total', 'index'], ascending=[False, True])

