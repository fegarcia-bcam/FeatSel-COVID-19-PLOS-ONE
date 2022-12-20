import os
import glob
import json

import numpy as np
import matplotlib.pyplot as plt

import config


IMG_FORMAT = 'tiff'
IMG_DPI = 300
IMG_EXTRA = {'compression': 'tiff_lzw'}

PATH = config.PATH_EXTRA_BIOINSP


if __name__ == '__main__':
    search = PATH + os.sep + '*.json'
    files = [os.path.basename(f) for f in glob.glob(search, recursive=False)]
    for file in files:
        file_path = os.path.join(PATH, file)
        with open(file_path, 'r') as f_in:
            result = json.load(f_in)
        fitness = result['fitness']

        analysis = result['analysis']
        feature_selector = result['feature_selector']
        estimator = result['selector_params']['estimator']

        if feature_selector == 'wrap-ga':
            n_pop = result['selector_params']['n_pop']
            mutxpb = result['selector_params']['mutxpb']
            title = r'GA-{} [{}]: $n_{{pop}}$: {:d}, $p_m$: {:0.3f}'.format(analysis.capitalize(), estimator, n_pop, mutxpb)

        elif feature_selector == 'wrap-bpso':
            v_c = result['selector_params']['velocity_clamp'][1]
            w = result['selector_params']['options']['w']
            iters = result['selector_params']['iters']
            title = r'BPSO-{} [{}]: |$v_{{max}}$|: {:d}, $\omega$: {:0.1f}'.format(analysis.capitalize(), estimator, v_c, w)

        else:
            raise RuntimeError

        fitness = list(map(list, zip(*fitness)))
        fig, ax = plt.subplots()
        for idx, fitn in enumerate(fitness):
            plt.plot(np.abs(fitn), label=str(idx))  # GA maximizes, BPSO minimizes
        ax.set_title(title)
        ax.legend(title='Bootstrap', ncol=2)
        plt.show(block=True)

        path_fig = os.path.join(config.PATH_EXTRA_BIOINSP, file.replace('json', IMG_FORMAT))
        fig.savefig(path_fig, format=IMG_FORMAT, dpi=IMG_DPI, pil_kwargs=IMG_EXTRA)
