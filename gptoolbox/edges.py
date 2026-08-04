"""edges.m -- unique undirected edges, sorted rows (as MATLAB returns them)."""

import numpy as np


def edges(F):
    F = np.asarray(F)
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], axis=0)
    E = np.sort(E, axis=1)
    return np.unique(E, axis=0)
