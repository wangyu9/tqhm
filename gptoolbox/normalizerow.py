"""normalizerow.m -- scale each row to unit l2 norm."""

import numpy as np

from normrow import normrow


def normalizerow(A):
    A = np.asarray(A, dtype=np.float64)
    return A / normrow(A)[:, None]
