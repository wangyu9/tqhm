"""cotmatrix.m -- cotangent Laplacian (negative semi-definite, as in gptoolbox).

Not part of the gptoolbox subset mirrored in `../qhm`; it lives in the full
gptoolbox that MATLAB has on its path, and ported files call it.

Cannot delegate to `igl.cotmatrix`: every sparse-returning binding in igl 2.5.1
here returns nnz == 0 (see gptoolbox/grad.py). Built from the standard cotangent
formula instead, and cross-checked against `grad` via `L = -G' diag(area) G`.
"""

import numpy as np
import scipy.sparse as sp

from edge_lengths import edge_lengths
from doublearea import doublearea


def cotmatrix(V, F):
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F)
    n = V.shape[0]

    # l[:, i] is the edge opposite corner i
    l2 = edge_lengths(V, F) ** 2
    dblA = doublearea(V, F)

    # cot at corner i = (l_j^2 + l_k^2 - l_i^2) / (4 * area)
    cot = np.stack([
        l2[:, 1] + l2[:, 2] - l2[:, 0],
        l2[:, 2] + l2[:, 0] - l2[:, 1],
        l2[:, 0] + l2[:, 1] - l2[:, 2],
    ], axis=1) / (2.0 * dblA[:, None])

    # corner i's cotangent weights the opposite edge (j,k)
    i = np.concatenate([F[:, 1], F[:, 2], F[:, 0]])
    j = np.concatenate([F[:, 2], F[:, 0], F[:, 1]])
    w = 0.5 * np.concatenate([cot[:, 0], cot[:, 1], cot[:, 2]])

    rows = np.concatenate([i, j, i, j])
    cols = np.concatenate([j, i, i, j])
    vals = np.concatenate([w, w, -w, -w])
    return sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
