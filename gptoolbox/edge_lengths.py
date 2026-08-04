"""edge_lengths.m -- per-face edge lengths, opposite-vertex ordering."""

import numpy as np
import igl


def edge_lengths(V, F):
    V = np.asarray(V, dtype=np.float64)
    if V.shape[1] == 2:
        V = np.c_[V, np.zeros(V.shape[0])]
    return igl.edge_lengths(V, np.asarray(F, dtype=np.int32))
