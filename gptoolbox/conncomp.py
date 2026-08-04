"""conncomp.m -- connected components of a graph given by its adjacency matrix.

`igl.connected_components` rejects a float adjacency matrix here and the sparse
igl bindings are unusable anyway (see gptoolbox/grad.py), so this uses scipy's
csgraph. Returns `(ncomp, C)` like gptoolbox's `[S, C]`.

Labels are **0-based**, unlike MATLAB's 1-based `C`. Callers that feed `C` into an
`accumarray`-style bincount need no shift, but a `C == longest` comparison must
use a 0-based `longest`.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components


def conncomp(A):
    A = sp.csr_matrix(A)
    ncomp, C = connected_components(A, directed=False)
    return ncomp, np.asarray(C, dtype=np.int64)
