"""attempt_local_global_old.m -- the original local/global fallback, before the
complex/cuDSS rewrite in attempt_local_global.m.

MATLAB script on the caller's workspace; here a function taking/returning the
variables it touches. It solves the real 2n x 2n anisotropic Laplace system with
a direct sparse solve (scipy's spsolve replaces MATLAB's backslash), which is
exactly what the newer version avoids.

Returns (u, v, num_flipped, a11, a12, a22).
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from tqhm_config import npy
from doublearea import doublearea


def _sparse_diag(ddd):
    d = np.asarray(ddd, dtype=np.float64).ravel()
    return sp.diags(d, 0, shape=(d.size, d.size), format='csr')


def attempt_local_global_old(mesh, da, tp, S, R, TVB, Area, F, f):
    # --- A local global solver: Initialization. ---
    au = npy(tp['s_at2au'](da.reshape(3, -1).t().contiguous())).reshape((f, 3), order='F')
    a11 = au[:, 0]
    a12 = au[:, 1]
    a22 = au[:, 2]

    GI = mesh['GI_sp'] if 'GI_sp' in mesh else mesh['GI']
    # Area = mesh['FA']

    u = v = None
    num_flipped = f

    for jj in range(1, 21):
        # global step:
        print('attempting local-global step\n')

        Lw = GI.T @ sp.bmat([[_sparse_diag(a11), _sparse_diag(a12)],
                             [_sparse_diag(a12), _sparse_diag(a22)]]) @ GI
        lhs = S.T @ Lw @ S
        rhs = -(S.T @ Lw @ R) @ TVB

        U = spla.spsolve(sp.csc_matrix(lhs), rhs)

        W = S @ U + R @ TVB

        u = W[:, 0]
        v = W[:, 1]

        np.trace(W.T @ (Lw @ W) / 2)   # value only; MATLAB echoed it

        # render meshes.
        # render_mesh2(W, F, EdgeColor=[0,0,0]); axis equal;

        # local step:
        if False:
            # the naive slow implementation:
            for j in range(f):
                Jj = (GI[[j, j + f], :] @ W).T
                Aj = abs(np.linalg.det(Jj)) * np.linalg.inv(Jj.T @ Jj) * Area[j]
                a11[j] = Aj[0, 0]
                a12[j] = Aj[0, 1]
                a22[j] = Aj[1, 1]
        else:
            # the fast implementation.
            GW = GI @ W
            Gxu = GW[:f, 0]
            Gxv = GW[:f, 1]
            Gyu = GW[f:2 * f, 0]
            Gyv = GW[f:2 * f, 1]
            # Jj = [[Gxu, Gxv], [Gyu, Gyv]].T

            adetJ = np.abs(Gxu * Gyv - Gxv * Gyu)

            a22 = (Gxu * Gxu + Gxv * Gxv) / adetJ * Area
            a12 = -(Gxu * Gyu + Gxv * Gyv) / adetJ * Area
            a11 = (Gyu * Gyu + Gyv * Gyv) / adetJ * Area

        newArea = doublearea(np.stack([u, v], axis=1), F)
        flipped = newArea < 0
        # number of flipped triangles.
        num_flipped = int(np.count_nonzero(flipped))
        print('flipps=%d, min_area=%g' % (num_flipped, newArea.min()), end='')

        if num_flipped == 0:
            break

    return u, v, num_flipped, a11, a12, a22
