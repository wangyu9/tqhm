"""linear_tensor_to_edge_weights.m -- per-face 3x3 map from the symmetric tensor
[a11, a12+a21, a22] to the three cotangent-style edge weights.

Row k pairs the hat-function gradients of two of the triangle's vertices, so
J[i] @ [a11; a12; a22] gives the off-diagonal Laplacian entries of face i.
"""

import numpy as np

from triangle_mesh import _basis_grad_core as _core


def linear_tensor_to_edge_weights(V, F):
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)

    g1 = _core(V, F[:, [0, 1, 2]])[:, :2]
    g2 = _core(V, F[:, [1, 2, 0]])[:, :2]
    g3 = _core(V, F[:, [2, 0, 1]])[:, :2]

    f = F.shape[0]

    J = np.zeros((f, 3, 3))

    for (a, b), k in zip([(g1, g2), (g2, g3), (g3, g1)], range(3)):
        J[:, k, 0] = a[:, 0] * b[:, 0]
        J[:, k, 1] = a[:, 0] * b[:, 1] + a[:, 1] * b[:, 0]
        J[:, k, 2] = a[:, 1] * b[:, 1]

    # optionally double check
    if False:
        A2w = J
        for i in range(f):
            assert abs(np.linalg.det(A2w[i])) > 1e-10

    return J
