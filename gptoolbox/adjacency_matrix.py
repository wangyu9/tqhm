"""adjacency_matrix.m -- symmetric vertex-vertex adjacency.

`igl.adjacency_matrix` returns nnz == 0 here (see gptoolbox/grad.py), so the
incidence pairs are built directly. `my_tutte` passes an edge list (#E by 2), not
just triangles, so both simplex sizes are handled.
"""

import numpy as np
import scipy.sparse as sp


def adjacency_matrix(F):
    F = np.asarray(F)
    n = int(F.max()) + 1
    ss = F.shape[1]

    if ss == 2:
        i, j = F[:, 0], F[:, 1]
    elif ss == 3:
        i = np.concatenate([F[:, 0], F[:, 1], F[:, 2]])
        j = np.concatenate([F[:, 1], F[:, 2], F[:, 0]])
    else:
        raise ValueError('unsupported simplex size %d' % ss)

    rows = np.concatenate([i, j])
    cols = np.concatenate([j, i])
    A = sp.coo_matrix((np.ones(rows.size), (rows, cols)), shape=(n, n)).tocsr()
    A.data[:] = 1.0   # gptoolbox returns a 0/1 pattern, not multiplicities
    return A
