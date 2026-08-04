"""facet_adjacency_matrix.m -- facets adjacent iff they share a (dim-1)-simplex.

igl only exposes triangle_triangle_adjacency, which cannot handle the edge lists
this is called with (my_tutte passes the boundary edges), so the incidence
product is built directly.
"""

import numpy as np
import scipy.sparse as sp


def facet_adjacency_matrix(F):
    F = np.asarray(F)
    nf, ss = F.shape
    n = int(F.max()) + 1

    if ss == 2:
        # facets of an edge are its two vertices
        S = F
    elif ss == 3:
        S = np.sort(np.concatenate([F[:, [1, 2]], F[:, [2, 0]], F[:, [0, 1]]], axis=0),
                    axis=1)
        _, S = np.unique(S, axis=0, return_inverse=True)
        S = S.reshape((3, nf)).T
        n = int(S.max()) + 1
    else:
        raise NotImplementedError('facet_adjacency_matrix.m: only edges/triangles')

    I = sp.coo_matrix(
        (np.ones(S.size), (np.repeat(np.arange(nf), S.shape[1]), S.ravel())),
        shape=(nf, n),
    ).tocsr()

    A = (I @ I.T).tolil()
    A.setdiag(0)
    return (A.tocsr() > 0).astype(np.float64)
