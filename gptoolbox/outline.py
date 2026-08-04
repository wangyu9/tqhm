"""outline.m -- boundary edges, preserving MATLAB's column-major find() order.

The ordering matters: mesh.IKB (the known/boundary vertex list) is built from
outline()'s first column, and downstream permutations depend on it.
"""

import numpy as np
import scipy.sparse as sp


def outline(F):
    F = np.asarray(F)
    n = int(F.max()) + 1
    ss = F.shape[1]
    Fnext = F[:, list(range(1, ss)) + [0]]

    A = sp.coo_matrix(
        (np.ones(F.size), (F.T.ravel(), Fnext.T.ravel())), shape=(n, n)
    ).tocsr()

    M = sp.coo_matrix(A - A.T)
    order = np.lexsort((M.row, M.col))  # MATLAB find(): column-major
    OI, OJ, OV = M.row[order], M.col[order], M.data[order]

    keep = OV > 0
    return np.stack([OI[keep], OJ[keep]], axis=1)
