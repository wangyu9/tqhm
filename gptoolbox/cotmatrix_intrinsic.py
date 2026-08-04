"""cotmatrix_intrinsic.m -- cotangent Laplacian from edge lengths alone.

Not delegated to `igl.cotmatrix_intrinsic` because every sparse-returning igl
binding here comes back with nnz == 0 (see gptoolbox/grad.py). Same formula as
cotmatrix.py, with the areas taken from Heron instead of the cross product.
"""

import numpy as np
import scipy.sparse as sp

from doublearea_intrinsic import doublearea_intrinsic


def cotmatrix_intrinsic(L, F, n=None):
    L = np.asarray(L, dtype=np.float64)
    F = np.asarray(F)
    if n is None:
        n = int(F.max()) + 1

    l2 = L ** 2
    dblA = doublearea_intrinsic(L)

    cot = np.stack([
        l2[:, 1] + l2[:, 2] - l2[:, 0],
        l2[:, 2] + l2[:, 0] - l2[:, 1],
        l2[:, 0] + l2[:, 1] - l2[:, 2],
    ], axis=1) / (2.0 * dblA[:, None])

    i = np.concatenate([F[:, 1], F[:, 2], F[:, 0]])
    j = np.concatenate([F[:, 2], F[:, 0], F[:, 1]])
    w = 0.5 * np.concatenate([cot[:, 0], cot[:, 1], cot[:, 2]])

    rows = np.concatenate([i, j, i, j])
    cols = np.concatenate([j, i, i, j])
    vals = np.concatenate([w, w, -w, -w])
    return sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
