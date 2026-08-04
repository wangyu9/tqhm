"""readOBJ.m -- read an OBJ file returning V, F, UV, TF.

Uses igl.read_obj. The test-case OBJs use "f v/vt" corners, so TF is read from
the texture indices; VT (the target boundary) comes from the vt lines.
"""

import numpy as np
import igl


def readOBJ(filename):
    V, UV, N, F, TF, NF = igl.read_obj(str(filename))
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    UV = np.asarray(UV, dtype=np.float64)
    TF = np.asarray(TF, dtype=np.int64)
    return V, F, UV, TF, np.asarray(N, dtype=np.float64), np.asarray(NF)
