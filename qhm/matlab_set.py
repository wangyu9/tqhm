"""MATLAB `intersect(...,'stable')` helpers (no MATLAB counterpart).

`[C,IA,IB] = intersect(A,B,'stable')` returns the common elements in order of
their first appearance in A. Everywhere it is used in this codebase the rows of A
are already unique and every row of A occurs in B, so the result reduces to "for
each row of A, its (first) index in B".
"""

import numpy as np


def intersect_stable(A, B):
    """Returns (C, IA, IB) for 1-D integer arrays, MATLAB 'stable' order."""
    A = np.asarray(A).ravel()
    B = np.asarray(B).ravel()

    b_first = {}
    for j, bv in enumerate(B):
        if bv not in b_first:
            b_first[bv] = j

    C, IA, IB = [], [], []
    seen = set()
    for i, av in enumerate(A):
        if av in b_first and av not in seen:
            seen.add(av)
            C.append(av)
            IA.append(i)
            IB.append(b_first[av])
    return np.array(C, dtype=A.dtype), np.array(IA, dtype=np.int64), np.array(IB, dtype=np.int64)


def intersect_stable_rows(A, B):
    """Row-wise variant of intersect_stable."""
    A = np.atleast_2d(np.asarray(A))
    B = np.atleast_2d(np.asarray(B))

    b_first = {}
    for j in range(B.shape[0]):
        key = tuple(B[j])
        if key not in b_first:
            b_first[key] = j

    rows, IA, IB = [], [], []
    seen = set()
    for i in range(A.shape[0]):
        key = tuple(A[i])
        if key in b_first and key not in seen:
            seen.add(key)
            rows.append(i)
            IA.append(i)
            IB.append(b_first[key])
    C = A[np.array(rows, dtype=np.int64)] if rows else A[:0]
    return C, np.array(IA, dtype=np.int64), np.array(IB, dtype=np.int64)
