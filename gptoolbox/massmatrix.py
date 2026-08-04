"""massmatrix.m -- diagonal FEM mass matrix ('voronoi' / 'barycentric').

Not part of the gptoolbox subset mirrored in `../qhm`; FEMpre.m calls it off
MATLAB's path. Cannot delegate to `igl.massmatrix`: every sparse-returning igl
binding here comes back with nnz == 0 (see gptoolbox/grad.py).

Only the diagonal types are implemented -- FEMpre.m asserts
`nnz(pre.Mass) == pre.n`, so 'full' would break it anyway.
"""

import numpy as np
import scipy.sparse as sp

from doublearea import doublearea
from edge_lengths import edge_lengths


def massmatrix(V, F, type='voronoi'):
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F)
    n = V.shape[0]
    assert F.shape[1] == 3, 'only triangle meshes'

    dblA = doublearea(V, F)

    if type == 'barycentric':
        MI = F.ravel(order='F')
        MV = np.tile(dblA / 6.0, 3)
    elif type == 'voronoi':
        l = edge_lengths(V, F)
        l2 = l ** 2
        cosines = np.stack([
            l2[:, 2] + l2[:, 1] - l2[:, 0],
            l2[:, 0] + l2[:, 2] - l2[:, 1],
            l2[:, 1] + l2[:, 0] - l2[:, 2],
        ], axis=1) / (2.0 * l[:, [1, 2, 0]] * l[:, [2, 0, 1]])
        barycentric = cosines * l
        normalized = barycentric / np.sum(barycentric, axis=1)[:, None]
        partial = normalized * (dblA / 2.0)[:, None]
        quads = np.stack([
            (partial[:, 1] + partial[:, 2]) / 2.0,
            (partial[:, 2] + partial[:, 0]) / 2.0,
            (partial[:, 0] + partial[:, 1]) / 2.0,
        ], axis=1)

        # obtuse triangles: the circumcenter falls outside, so split by area
        obtuse0 = cosines[:, 0] < 0
        quads[obtuse0, :] = (dblA[obtuse0] / 4.0)[:, None] * np.array([1.0, 0.5, 0.5])
        obtuse1 = cosines[:, 1] < 0
        quads[obtuse1, :] = (dblA[obtuse1] / 4.0)[:, None] * np.array([0.5, 1.0, 0.5])
        obtuse2 = cosines[:, 2] < 0
        quads[obtuse2, :] = (dblA[obtuse2] / 4.0)[:, None] * np.array([0.5, 0.5, 1.0])

        MI = F.ravel(order='F')
        MV = quads.ravel(order='F')
    else:
        raise NotImplementedError('unsupported mass matrix type: %s' % type)

    return sp.coo_matrix((MV, (MI, MI)), shape=(n, n)).tocsr()
