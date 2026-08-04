"""grad_vertex_over_edge.m -- one right inverse of d(edge)/d(vertex).

dv/de is not unique; `_core` picks the solution that moves only the vertex opposite
the edge. The MATLAB file initializes `dvode` with ones and only overwrites the two
columns each `_core` call touches, so the untouched entries stay 1; that is
reproduced here rather than zeroed.
"""

import numpy as np


def grad_vertex_over_edge(VX):
    """dvode: 3 x 6, one possibility of how edge velocity determines vertex velocity."""
    VX = np.asarray(VX, dtype=np.float64)

    dvode = np.ones((3, 6))
    dvode[2, [0, 1, 2, 3, 4, 5]] = _core(VX[[0, 1, 2], :])
    dvode[0, [2, 3, 4, 5, 0, 1]] = _core(VX[[1, 2, 0], :])
    dvode[1, [4, 5, 0, 1, 2, 3]] = _core(VX[[2, 0, 1], :])

    if False:
        ge = np.ones(3)
        ge[2] = _value(gv[[0, 1, 2], :], VX[[0, 1, 2], :])
        ge[0] = _value(gv[[1, 2, 0], :], VX[[1, 2, 0], :])
        ge[1] = _value(gv[[2, 0, 1], :], VX[[2, 0, 1], :])

    return dvode


def _rn(x):
    return x / np.linalg.norm(x)


def _core(VX):
    # 1,2,3 is assumed counter-clockwise, but the formula also holds for an
    # inverted triangle, where negative angles pop up.
    assert VX.shape[1] == 2

    v32 = VX[1, :] - VX[2, :]
    v32R = np.array([-v32[1], v32[0]])

    v12 = VX[1, :] - VX[0, :]

    v32RN = _rn(v32R)
    v12N = _rn(v12)
    v32N = _rn(v32)

    # negative if the triangle indexing is not counter-clockwise;
    # sinangle2 = sin(<021 - <023)
    sinangle2 = v12N[1] * v32N[0] - v12N[0] * v32N[1]

    # de/dv1 / (de/dl) = dl/dv1 = sin <123, so de/dv1 = g2 .* v32RN
    return np.concatenate([[0, 0], v32RN / sinangle2, [0, 0]])


def _value(gv, VX):
    assert VX.shape[1] == 2

    g2 = gv[1, :]

    v32 = VX[1, :] - VX[2, :]
    v32R = np.array([-v32[1], v32[0]])

    v12 = VX[1, :] - VX[0, :]

    v32RN = _rn(v32R)
    v12N = _rn(v12)
    v32N = _rn(v32)

    sinangle2 = v12N[1] * v32N[0] - v12N[0] * v32N[1]

    # (v12N .* v32RN) is cos of the angle <123.
    return g2 * v32RN / sinangle2
