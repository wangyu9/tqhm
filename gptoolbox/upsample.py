"""upsample.m -- one round of 1-to-4 triangle subdivision."""

import numpy as np
import igl


def upsample(V, F):
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int32)
    pad = V.shape[1] == 2
    if pad:
        V = np.c_[V, np.zeros(V.shape[0])]
    NV, NF = igl.upsample(V, F)
    if pad:
        NV = NV[:, :2]
    return np.asarray(NV, dtype=np.float64), np.asarray(NF, dtype=np.int64)
