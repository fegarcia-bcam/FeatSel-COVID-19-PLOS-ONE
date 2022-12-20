import numpy as np
import pyswarms as ps

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from imblearn.metrics import geometric_mean_score

from feature_selection_ga import FeatureSelectionGA, FitnessFunction

import config


# Genetic Algorithms
def ga(X, Y, alp, n_sp, estimator, n_pop, cxpb, mutxpb, ngen):
    class FitnessFunction:
        def __init__(self, n_feats_tot, alpha, nsplits, *args, **kwargs):
            self.alpha = alpha
            self.n_feats_tot = n_feats_tot
            self.nsplits = nsplits

        def calculate_fitness(self, model, x, y):
            cv_set = np.repeat(-1., x.shape[0])
            n_feats = x.shape[1]

            skf = StratifiedKFold(n_splits=self.nsplits, shuffle=True)
            for train_index, test_index in skf.split(x, y):
                x_train, x_test = x[train_index], x[test_index]
                y_train, y_test = y[train_index], y[test_index]
                if x_train.shape[0] != y_train.shape[0]:
                    raise RuntimeError

                mod = clone(model)
                mod.fit(x_train, y_train)
                y_pred = mod.predict(x_test)
                cv_set[test_index] = y_pred

            y_pred = np.around(cv_set)
            y_pred[y_pred < 0] = 0
            y_pred[y_pred > config.NUM_CLASSES - 1] = config.NUM_CLASSES - 1
            sc_classif = geometric_mean_score(Y, y_pred)
            j = (self.alpha * sc_classif + (1.0 - self.alpha) * (1 - (n_feats / self.n_feats_tot)))
            return j

    ff = FitnessFunction(n_feats_tot=X.shape[1], alpha=alp, nsplits=n_sp)
    fsga = FeatureSelectionGA(estimator, X, Y, ff_obj=ff)
    fsga.generate(n_pop, cxpb, mutxpb, ngen)

    pos = fsga.best_ind
    pos_sel = np.where(pos)[0]
    X_sel_feats = X[:, pos_sel]
    return X_sel_feats, np.array(pos)


# BPSO
def bpso(X, Y, alp, n_sp, estimator, n_particles, iters, options, velocity_clamp):
    dim = np.shape(X)[1]

    def f_aux(m, alpha, nsplits):
        n_feats_tot = dim
        if np.count_nonzero(m) == 0:
            X_subset = X
        else:
            X_subset = X[:, m == 1]
        cv_set = np.repeat(-1., X.shape[0])
        n_feats = X_subset.shape[1]

        skf = StratifiedKFold(n_splits=nsplits, shuffle=True)
        for train_index, test_index in skf.split(X_subset, Y):
            x_train, x_test = X_subset[train_index], X_subset[test_index]
            y_train, y_test = Y[train_index], Y[test_index]
            if x_train.shape[0] != y_train.shape[0]:
                raise RuntimeError

            est = clone(estimator)
            est.fit(x_train, y_train)
            y_pred = est.predict(x_test)
            cv_set[test_index] = y_pred

        y_pred = np.around(cv_set)
        y_pred[y_pred < 0] = 0
        y_pred[y_pred > config.NUM_CLASSES - 1] = config.NUM_CLASSES - 1
        sc_classif = geometric_mean_score(Y, y_pred)
        j = -(alpha * sc_classif + (1.0 - alpha) * (1 - (n_feats / n_feats_tot)))
        return j

    def f(x, alpha=alp, nsplits=n_sp):
        # n_particles = x.shape[0]
        j = [f_aux(m=part, alpha=alpha, nsplits=nsplits) for part in x]
        return np.array(j)

    optimizer = ps.discrete.BinaryPSO(n_particles=n_particles, dimensions=dim, options=options,
                                      velocity_clamp=velocity_clamp)
    cost, pos = optimizer.optimize(f, iters=iters)

    X_sel_feats = X[:, pos == 1]
    return X_sel_feats, pos
