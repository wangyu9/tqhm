"""assemble_lap.m -- one-shot assembly of the anisotropic Laplacian.

The older, non-reusable counterpart of assemble_lap_core.py: it rebuilds the
sparsity pattern on every call. MATLAB's `sortrows([JJ,II])` + `sparse2` exists
only to hand `sparse2` pre-sorted triplets; scipy's coo->csr conversion sums
duplicates itself, so the sort is dropped and the (I,J,V) triplets go straight
into a coo_matrix.

Returns scipy CSR matrices (MATLAB returns sparse); a11/a12/a22 may be torch
tensors or numpy arrays.
"""

import numpy as np
import scipy.sparse as sp

from tqhm_config import npy


def assemble_lap(n, GIS, F, a11, a12, a22):
    A_lower = None
    A_full = None

    g1 = npy(GIS['g1'])
    g2 = npy(GIS['g2'])
    g3 = npy(GIS['g3'])

    a11 = npy(a11).ravel()
    a12 = npy(a12).ravel()
    a22 = npy(a22).ravel()

    F = np.asarray(F)

    output_full = True
    output_lower = True

    if output_full:
        II = np.concatenate([F[:, 0], F[:, 1], F[:, 2],
                             F[:, 0], F[:, 1], F[:, 2],
                             F[:, 1], F[:, 2], F[:, 0]])
        JJ = np.concatenate([F[:, 0], F[:, 1], F[:, 2],
                             F[:, 1], F[:, 2], F[:, 0],
                             F[:, 0], F[:, 1], F[:, 2]])

    if output_lower:
        maxFF12 = np.maximum(F[:, 0], F[:, 1])
        minFF12 = np.minimum(F[:, 0], F[:, 1])

        maxFF23 = np.maximum(F[:, 1], F[:, 2])
        minFF23 = np.minimum(F[:, 1], F[:, 2])

        maxFF31 = np.maximum(F[:, 2], F[:, 0])
        minFF31 = np.minimum(F[:, 2], F[:, 0])

        II_lower = np.concatenate([F[:, 0], F[:, 1], F[:, 2],
                                   maxFF12, maxFF23, maxFF31])
        JJ_lower = np.concatenate([F[:, 0], F[:, 1], F[:, 2],
                                   minFF12, minFF23, minFF31])

    def inner_prod(ss, tt):
        return (ss[:, 0] * a11 * tt[:, 0]
                + a12 * (ss[:, 0] * tt[:, 1] + ss[:, 1] * tt[:, 0])
                + ss[:, 1] * a22 * tt[:, 1])

    # v12=v21, v23=v32, v31=v13
    v12 = inner_prod(g1, g2)
    v23 = inner_prod(g2, g3)
    v31 = inner_prod(g3, g1)
    # no need to multiple with mesh.AI again. already did so.

    v11 = -(v12 + v31)
    v22 = -(v23 + v12)
    v33 = -(v31 + v23)

    if output_full:
        VV = np.concatenate([v11, v22, v33, v12, v23, v31, v12, v23, v31])
        A_full = sp.coo_matrix((VV, (II, JJ)), shape=(n, n)).tocsr()

    if output_lower:
        VV_lower = np.concatenate([v11, v22, v33, v12, v23, v31])
        A_lower = sp.coo_matrix((VV_lower, (II_lower, JJ_lower)),
                                shape=(n, n)).tocsr()

    return A_lower, A_full
