"""signed_mean_value_laplacian.m -- mean-value Laplacian with *signed* half-angle
tangents, so it stays well defined on flipped triangles.

The same weights triangle_mesh.m assembles into mesh.MVL; kept as a standalone
function because that is how the MATLAB source has it.
"""

import numpy as np
import scipy.sparse as sp

from edge_lengths import edge_lengths


def signed_mean_value_laplacian(V, F):
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)

    n = V.shape[0]
    f = F.shape[0]

    mesh = {'V': V, 'F': F}

    EL = edge_lengths(V, F)
    mesh['EL'] = EL

    v12 = V[F[:, 1], :] - V[F[:, 0], :]
    v13 = V[F[:, 2], :] - V[F[:, 0], :]

    v23 = V[F[:, 2], :] - V[F[:, 1], :]
    v21 = V[F[:, 0], :] - V[F[:, 1], :]

    v31 = V[F[:, 0], :] - V[F[:, 2], :]
    v32 = V[F[:, 1], :] - V[F[:, 2], :]

    dot23 = np.sum(v12 * v13, axis=1)
    dot31 = np.sum(v23 * v21, axis=1)
    dot12 = np.sum(v31 * v32, axis=1)

    mesh['dot23'] = dot23
    mesh['dot31'] = dot31
    mesh['dot12'] = dot12

    # 2D cross
    cross23 = v12[:, 0] * v13[:, 1] - v12[:, 1] * v13[:, 0]
    cross31 = v23[:, 0] * v21[:, 1] - v23[:, 1] * v21[:, 0]
    cross12 = v31[:, 0] * v32[:, 1] - v31[:, 1] * v32[:, 0]

    mesh['cross23'] = cross23
    mesh['cross31'] = cross31
    mesh['cross12'] = cross12

    # the *signed* tan of half the angle 213 (right hand rule on triangle 123 for
    # the positive sign): tan(theta/2) = sin(theta) / (1+cos(theta))
    stan1 = cross23 / (dot23 + EL[:, 1] * EL[:, 2])
    stan2 = cross31 / (dot31 + EL[:, 2] * EL[:, 0])
    stan3 = cross12 / (dot12 + EL[:, 0] * EL[:, 1])

    stan = np.stack([stan1, stan2, stan3], axis=1)
    mesh['stan'] = stan

    VV = np.concatenate([stan / EL[:, [2, 0, 1]], stan / EL[:, [1, 2, 0]]], axis=1)
    II = np.concatenate([F, F], axis=1)
    JJ = np.concatenate([F[:, [1, 2, 0]], F[:, [2, 0, 1]]], axis=1)

    # MVL = sparse(II,JJ,VV,n,n); MVL = MVL - diag(sum(MVL,2)); equivalently:
    vv = VV.ravel(order='F')
    ii = II.ravel(order='F')
    jj = JJ.ravel(order='F')
    MVL = (sp.coo_matrix((vv, (ii, jj)), shape=(n, n))
           - sp.coo_matrix((vv, (ii, ii)), shape=(n, n))).tocsr()

    # sum(sum(abs(MVL - mean_value_laplacian(V,F)))) gives a number around 0
    return MVL
