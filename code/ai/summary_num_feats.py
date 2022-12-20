import os
import glob
import json

import numpy as np
import pandas as pd

import config


if __name__ == '__main__':
    paths = [config.PATH_RESULTS_FIL, config.PATH_RESULTS_WRA, config.PATH_RESULTS_EMB]

    quants = [0.5, config.ALPHA / 2.0,  1.0 - config.ALPHA / 2.0]
    l_num_feats = []
    for path in paths:
        search = path + '/**/' + '* NFS no.json'
        for file_in in glob.glob(search, recursive=True):
            info = os.path.basename(file_in)
            info, _ = os.path.splitext(info)
            _, name = info.split('_', maxsplit=1)
            d_num_feats = {'name': name}

            with open(file_in, 'r') as f_in:
                results = json.load(f_in)

            Z = results['sel_result']
            Z = np.asarray(Z)
            num_feats = np.sum(Z, axis=1)
            nf_quants = np.quantile(num_feats, q=quants)

            for q, nf_q in zip(quants, nf_quants):
                d_num_feats.update({'q_{:0.3f}'.format(q): nf_q})

            l_num_feats.append(d_num_feats)

    df_num_feats = pd.DataFrame.from_dict(l_num_feats)
