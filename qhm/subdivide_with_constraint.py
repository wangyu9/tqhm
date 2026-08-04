"""subdivide_with_constraint.m -- subdivide (V,F) while carrying the linear
constraints eq_lhs * [x;y] == eq_rhs onto the refined mesh.

The prolongation matrix M is read off by subdividing `[V speye(n)]`, so the
weights columns ride along with the positions -- which is why
upsample_with_weights accepts a scipy sparse V. A constrained vertex of the fine
mesh is one whose row of M*eq_lhs' sums to exactly 1, i.e. it interpolates a
single constrained coarse vertex (a midpoint of two constrained vertices sums to
1 as well and is kept, which is the intent).

`find(...)` on a sparse column is column-major, but a single column is the same
either way here.
"""

import numpy as np
import scipy.sparse as sp

from upsample_with_weights import upsample_with_weights


def subdivide_with_constraint(V, F, eq_lhs, eq_rhs, num_iters):
    V2 = V
    F2 = F
    eq_lhs2 = sp.csr_matrix(eq_lhs)
    eq_rhs2 = np.asarray(eq_rhs, dtype=np.float64)

    for _ in range(1, int(num_iters) + 1):
        V_old = np.asarray(V2, dtype=np.float64)
        F_old = np.asarray(F2, dtype=np.int64)
        eq_lhs_old = eq_lhs2
        eq_rhs_old = eq_rhs2

        VVB, FF, _, _ = upsample_with_weights(
            sp.hstack([sp.csr_matrix(V_old), sp.eye(V_old.shape[0], format='csr')],
                      format='csr'),
            F_old)   # 'OnlySelected', invTri

        VV = np.asarray(VVB[:, :V_old.shape[1]].todense())
        M = sp.csr_matrix(VVB[:, V_old.shape[1]:])
        # max(max(abs(VV - M*V_old)))  -- MATLAB prints this as a sanity check
        _ = np.abs(VV - M @ V_old).max()

        MM = sp.block_diag([M, M], format='csr')

        fc = np.flatnonzero(np.asarray((MM @ eq_lhs_old.T).sum(axis=1)).ravel() == 1)

        eq_lhs_new = sp.coo_matrix(
            (np.ones(fc.size), (np.arange(fc.size), fc)),
            shape=(fc.size, 2 * VV.shape[0])).tocsr()
        eq_rhs_new = eq_lhs_new @ MM @ eq_lhs_old.T @ eq_rhs_old

        V2 = VV
        F2 = FF
        eq_lhs2 = eq_lhs_new
        eq_rhs2 = eq_rhs_new

    return V2, F2, eq_lhs2, eq_rhs2
