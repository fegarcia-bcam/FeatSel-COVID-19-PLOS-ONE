import numpy as np
import scipy.stats as st
import statistics as stats

from sklearn.metrics import jaccard_score

from stability import hypothesisTestV, getVarianceofStability, confidenceIntervals
import config


def test_higher(Z, val_thr):
    hypoth_test = hypothesisTestV(Z, stab0=val_thr, alpha=config.ALPHA)

    return hypoth_test


def get_similarities(Z1, Z2, method=None):
    if method is None:
        method = config.SIMILARITY_METHOD

    if method != 'jaccard':
        raise NotImplementedError

    jac_idx = []
    for s1, s2 in zip(Z1, Z2):
        jac_idx.append(jaccard_score(s1, s2))

    jac_mean = stats.mean(jac_idx)
    jac_median = stats.median(jac_idx)
    jac_std = stats.pstdev(jac_idx)

    return jac_idx, jac_mean, jac_median, jac_std


def get_stability_stats(result):
    Z = result['sel_result']

    stab_stats = getVarianceofStability(Z)
    stab_mean = stab_stats['stability']
    stab_var = stab_stats['variance']

    stab_ci = confidenceIntervals(Z, alpha=config.ALPHA, res=stab_stats)
    stab_ci_up = stab_ci['upper']
    stab_ci_lo = stab_ci['lower']
    stab_ci_d = (stab_ci_up - stab_ci_lo) / 2.0  # symmetrical

    selected = []
    for sel in Z:
        selected.append(sum(sel))
    sel_mean = stats.mean(selected)
    sel_median = stats.median(selected)
    sel_std = stats.pstdev(selected)

    return stab_mean, stab_var, stab_ci_d, sel_mean, sel_median, sel_std


def get_time_stats(result):
    times = result['time']
    t_mean = np.mean(times)
    t_ci_lo, t_ci_up = st.norm.interval(alpha=0.95, loc=t_mean, scale=st.sem(times))
    t_ci_d = (t_ci_up - t_ci_lo) / 2.0  # symmetrical

    return t_ci_lo, t_ci_d


def count_occurrences(Z, feat_names=None):
    Z = np.asarray(Z)
    m, d = Z.shape
    if feat_names is None:
        feat_names = [str(i) for i in range(d)]

    occur = np.count_nonzero(Z, axis=0).tolist()
    feat_occur = dict(zip(feat_names, occur))

    return feat_occur


def filter_by_occur(occur, thr):
    filt_occur = dict()
    for idx, val in enumerate(occur.values()):
        if val >= thr:
            filt_occur[idx] = val

    return filt_occur
