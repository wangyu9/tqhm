"""symbolic_lap.m -- sparsity pattern of the Laplacian (values are all 1)."""

import numpy as np
import scipy.sparse as sp


def symbolic_lap(n, F):
    F = np.asarray(F)
    II = np.concatenate([F[:, 0], F[:, 1], F[:, 2], F[:, 0], F[:, 1], F[:, 2],
                         F[:, 1], F[:, 2], F[:, 0]])
    JJ = np.concatenate([F[:, 0], F[:, 1], F[:, 2], F[:, 1], F[:, 2], F[:, 0],
                         F[:, 0], F[:, 1], F[:, 2]])
    return sp.coo_matrix((np.ones(II.size), (II, JJ)), shape=(n, n)).tocsr()
