"""grad_edge_over_vertex.m -- d(edge length)/d(vertex) for one triangle.

Same layout as grad_edgesq_over_vertex, but for the edge length itself. The unused
local `value` subfunction of the MATLAB file is kept as _value; it applies the same
chain rule to an intrinsic gradient.
"""

import numpy as np


def grad_edge_over_vertex(VX):
    """deodv: 6 x 3, how vertex velocity determines the edge velocity, de/dv."""
    VX = np.asarray(VX, dtype=np.float64)

    deodv = np.zeros((6, 3))

    deodv[np.ix_([0, 1], [0, 1, 2])] = _core(VX[[0, 1, 2], :])
    deodv[np.ix_([2, 3], [1, 2, 0])] = _core(VX[[1, 2, 0], :])
    deodv[np.ix_([4, 5], [2, 0, 1])] = _core(VX[[2, 0, 1], :])

    if False:
        # gv: 6 x 1
        gv = np.concatenate([
            _value(ge[[0, 1, 2]], VX[[0, 1, 2], :]),
            _value(ge[[1, 2, 0]], VX[[1, 2, 0], :]),
            _value(ge[[2, 0, 1]], VX[[2, 0, 1], :]),
        ])

    return deodv


def _core(VX):
    # deodv1: 2 x 3
    assert VX.shape[1] == 2

    v12 = VX[1, :] - VX[0, :]
    v31 = VX[0, :] - VX[2, :]

    e3 = np.linalg.norm(v12)
    e2 = np.linalg.norm(v31)

    # e2 is unchanged so no effect.
    return np.stack([np.zeros(2), v31 / e2, -v12 / e3], axis=1)


def _value(ge, VX):
    # gv1: 2 x 1
    ge = np.asarray(ge, dtype=np.float64).ravel()
    assert ge.size == 3
    assert VX.shape[1] == 2

    v12 = VX[1, :] - VX[0, :]
    v31 = VX[0, :] - VX[2, :]

    e3 = np.linalg.norm(v12)
    e2 = np.linalg.norm(v31)

    # e2 is unchanged so no effect.
    return ge[2] * (-v12) / e3 + ge[1] * v31 / e2
