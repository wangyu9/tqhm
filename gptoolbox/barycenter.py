"""barycenter.m -- barycenter of every face."""

import numpy as np
import igl


def barycenter(V, F):
    return igl.barycenter(np.asarray(V, dtype=np.float64), np.asarray(F, dtype=np.int32))
