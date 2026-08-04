"""vertex_facet_adjacency.m -- n x f 0/1 incidence of vertices in facets.

Not part of the gptoolbox subset mirrored in `../qhm`; it lives in the full
gptoolbox that MATLAB has on its path, and FEMpre.m calls it. igl has no
equivalent binding, so it is built directly.
"""

import numpy as np
import scipy.sparse as sp


def vertex_facet_adjacency(F, n=None):
    F = np.asarray(F)
    f, ss = F.shape
    if n is None:
        n = int(F.max()) + 1

    rows = F.ravel(order='F')
    cols = np.tile(np.arange(f), ss)
    J = sp.coo_matrix((np.ones(rows.size), (rows, cols)), shape=(n, f)).tocsr()
    J.data[:] = 1.0     # a vertex appearing twice in a facet still counts once
    return J
