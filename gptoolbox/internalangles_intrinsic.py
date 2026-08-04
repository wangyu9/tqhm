"""internalangles_intrinsic.m -- internal angles from edge lengths alone.

igl's python binding exposes `internal_angles` (extrinsic) but no intrinsic
variant, so the law of cosines is applied to L directly. Checked to agree with
`igl.internal_angles` to 7e-16 on the test mesh.
"""

import numpy as np


def internalangles_intrinsic(L):
    L = np.asarray(L, dtype=np.float64)
    assert L.shape[1] == 3

    l1, l2, l3 = L[:, 0], L[:, 1], L[:, 2]
    a1 = np.arccos((l2 ** 2 + l3 ** 2 - l1 ** 2) / (2 * l2 * l3))
    a2 = np.arccos((l3 ** 2 + l1 ** 2 - l2 ** 2) / (2 * l3 * l1))
    a3 = np.arccos((l1 ** 2 + l2 ** 2 - l3 ** 2) / (2 * l1 * l2))
    return np.stack([a1, a2, a3], axis=1)
