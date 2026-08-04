"""grad_edgesq_over_vertex.m -- d(edge length squared)/d(vertex) for one triangle.

Single-triangle helper (VX is 3x2), so it stays in numpy; the vectorized form used
in the hot path is batch_grad_edgesq_over_vertex.
"""

import numpy as np


def grad_edgesq_over_vertex(VX):
    VX = np.asarray(VX, dtype=np.float64)

    deodv = np.zeros((6, 3))

    deodv[np.ix_([0, 1], [0, 1, 2])] = _core(VX[[0, 1, 2], :])
    deodv[np.ix_([2, 3], [1, 2, 0])] = _core(VX[[1, 2, 0], :])
    deodv[np.ix_([4, 5], [2, 0, 1])] = _core(VX[[2, 0, 1], :])

    return deodv


def _core(VX):
    # deodv1: 2 x 3
    assert VX.shape[1] == 2

    v12 = VX[1, :] - VX[0, :]
    v31 = VX[0, :] - VX[2, :]

    # e2 is unchanged so no effect.
    return 2 * np.stack([np.zeros(2), v31, -v12], axis=1)
