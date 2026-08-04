"""grad.m -- numerical gradient operator for a triangle mesh.

Deviates from the "delegate to igl" rule out of necessity: every sparse-returning
binding in igl 2.5.1 here (`grad`, `cotmatrix`, `massmatrix`) hands back a matrix
with nnz == 0, so `igl.grad` cannot be used. The MATLAB formulas are transcribed
instead. igl's dense returns (`doublearea`, `edge_lengths`, ...) are unaffected,
so the other gptoolbox wrappers still delegate.
"""

import numpy as np
import scipy.sparse as sp

from normrow import normrow
from normalizerow import normalizerow


def grad(V, F):
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F)
    if V.ndim == 1:
        V = V[:, None]
    dim = V.shape[1]
    ss = F.shape[1]
    m, nv = F.shape[0], V.shape[0]

    if ss == 2:
        # staggered-grid finite difference along each edge
        length = np.abs(V.ravel()[F[:, 1]] - V.ravel()[F[:, 0]])
        rows = np.repeat(np.arange(m), 2)
        cols = F.ravel()
        vals = np.stack([1.0 / length, -1.0 / length], axis=1).ravel()
        return sp.coo_matrix((vals, (rows, cols)), shape=(m, nv)).tocsr()

    if ss == 3:
        if dim == 2:
            V = np.c_[V, np.zeros(nv)]

        i1, i2, i3 = F[:, 0], F[:, 1], F[:, 2]
        v32 = V[i3, :] - V[i2, :]
        v13 = V[i1, :] - V[i3, :]
        v21 = V[i2, :] - V[i1, :]

        n = np.cross(v32, v13)
        dblA = normrow(n)          # twice the triangle area
        u = normalizerow(n)

        # each edge vector rotated 90 degrees about the face normal
        eperp21 = np.cross(u, v21) / dblA[:, None]
        eperp13 = np.cross(u, v13) / dblA[:, None]

        idx = np.arange(m)
        rows = np.concatenate([d * m + np.tile(idx, 4) for d in range(3)])
        cols = np.tile(np.concatenate([F[:, 1], F[:, 0], F[:, 2], F[:, 0]]), 3)
        vals = np.concatenate([
            np.concatenate([eperp13[:, d], -eperp13[:, d],
                            eperp21[:, d], -eperp21[:, d]])
            for d in range(3)
        ])

        G = sp.coo_matrix((vals, (rows, cols)), shape=(3 * m, nv)).tocsr()
        if dim == 2:
            G = G[:m * dim, :]
        return G

    if ss == 4:
        # needs normals() and volume(), which this port never mirrors
        raise NotImplementedError(
            'tet gradient needs gptoolbox normals/volume, not ported')

    raise ValueError('unsupported simplex size %d' % ss)
