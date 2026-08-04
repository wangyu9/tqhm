"""doublearea_intrinsic.m -- twice the triangle area from edge lengths.

The igl python binding has no `doublearea_intrinsic`, so Heron's formula is
applied in the numerically stable (sorted) form gptoolbox/libigl use.
"""

import numpy as np


def doublearea_intrinsic(L):
    L = np.asarray(np.atleast_2d(L), dtype=np.float64)
    assert L.shape[1] == 3

    # sort descending so the stable form of Heron's formula applies
    S = -np.sort(-L, axis=1)
    a, b, c = S[:, 0], S[:, 1], S[:, 2]
    return 0.5 * np.sqrt(
        np.maximum((a + (b + c)) * (c - (a - b)) * (c + (a - b)) * (a + (b - c)), 0.0)
    )
