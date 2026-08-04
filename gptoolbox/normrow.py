"""normrow.m -- l2 norm of each row."""

import numpy as np


def normrow(A):
    return np.sqrt(np.sum(np.asarray(A, dtype=np.float64) ** 2, axis=1))
