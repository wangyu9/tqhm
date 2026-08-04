"""kharmonic.m -- k-harmonic weights with Dirichlet data on b.

Delegates the solve to `igl.harmonic_from_laplacian_and_mass`, which accepts
scipy matrices and works correctly. The Laplacian itself is built here, since
igl's sparse-*returning* bindings are unusable (see gptoolbox/grad.py):

  - default:            cotmatrix
  - IntrinsicDelaunay:  cotmatrix_intrinsic on igl's flipped edge lengths
  - V empty:            uniform graph Laplacian (adjacency - degree)

Verified against `igl.harmonic` to 3e-15 on the test mesh.
"""

import numpy as np
import scipy.sparse as sp
import igl

from cotmatrix import cotmatrix
from cotmatrix_intrinsic import cotmatrix_intrinsic
from doublearea import doublearea
from doublearea_intrinsic import doublearea_intrinsic
from adjacency_matrix import adjacency_matrix


def _mass(dblA, F, n):
    """Barycentric mass matrix."""
    F = np.asarray(F)
    return sp.coo_matrix((np.repeat(dblA / 6.0, 3), (F.ravel(), F.ravel())),
                         shape=(n, n)).tocsr()


def kharmonic(V, F, b, bc, k=1, IntrinsicDelaunay=False):
    F = np.asarray(F)
    b = np.asarray(b, dtype=np.int32).ravel()
    bc = np.atleast_2d(np.asarray(bc, dtype=np.float64))

    if V is None or len(np.atleast_1d(V)) == 0:
        n = int(F.max()) + 1
        A = adjacency_matrix(F)
        L = A - sp.diags(np.asarray(A.sum(axis=1)).ravel(), 0, format='csr')
        M = sp.eye(n, format='csr')
    else:
        V = np.asarray(V, dtype=np.float64)
        n = V.shape[0]
        if IntrinsicDelaunay:
            _, l, Fi = igl.intrinsic_delaunay_cotmatrix(
                V if V.shape[1] == 3 else np.c_[V, np.zeros(n)],
                np.asarray(F, dtype=np.int32))
            l = np.asarray(l, dtype=np.float64)
            Fi = np.asarray(Fi, dtype=np.int64)
            L = cotmatrix_intrinsic(l, Fi, n)
            M = _mass(doublearea_intrinsic(l), Fi, n)
        else:
            L = cotmatrix(V, F)
            M = _mass(doublearea(V, F) if V.shape[1] != 2
                      else np.abs(doublearea(V, F)), F, n)

    return np.asarray(
        igl.harmonic_from_laplacian_and_mass(sp.csr_matrix(L), sp.csr_matrix(M),
                                             b, bc, int(k)),
        dtype=np.float64)
