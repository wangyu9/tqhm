"""assemble_lap_core.m -- precompute the sparsity structure of the anisotropic
Laplacian once, then rebuild only its values on every iteration.

MATLAB relies on sparse(I,J,V) summing duplicate entries. Here the (I,J) pairs
are sorted once (column-major, as MATLAB does) and duplicates are collapsed into
a fixed CSR layout, so each rebuild is a gather + segment-sum with no
re-sorting. That fixed layout is also what lets the cuDSS symbolic
factorization be reused across iterations.
"""

import torch

from tqhm_config import DEV, DT, ti


class _Assembler:
    """Fixed sparsity pattern; `values()` recomputes the CSR data array."""

    def __init__(self, n, II, JJ, tri_of_entry, is_full):
        self.n = n
        self.is_full = is_full

        # Collapse duplicate (row, col) pairs into a canonical CSR ordering.
        lin = II * n + JJ
        uniq, inverse = torch.unique(lin, return_inverse=True)
        self.nnz = int(uniq.numel())

        rows = uniq // n
        cols = uniq % n

        counts = torch.bincount(rows + 1, minlength=n + 1)
        self.indptr = torch.cumsum(counts, 0).to(torch.int64)
        self.indices = cols.to(torch.int64)

        # scatter_add target for each of the 9 (or 6) blocks of f entries
        self.inverse = inverse.to(device=DEV, dtype=torch.int64)
        self.tri_of_entry = tri_of_entry

    def values(self, v11, v22, v33, v12, v23, v31):
        if self.is_full:
            VV = torch.cat([v11, v22, v33, v12, v23, v31, v12, v23, v31])
        else:
            VV = torch.cat([v11, v22, v33, v12, v23, v31])
        data = torch.zeros(self.nnz, dtype=DT, device=DEV)
        data.scatter_add_(0, self.inverse, VV)
        return data


def _inner_prod_values(GIS, a11, a12, a22):
    """The six distinct per-triangle tensor-weighted basis inner products."""
    g1, g2, g3 = GIS['g1'], GIS['g2'], GIS['g3']

    def inner(s, t):
        return (s[:, 0] * a11 * t[:, 0]
                + a12 * (s[:, 0] * t[:, 1] + s[:, 1] * t[:, 0])
                + s[:, 1] * a22 * t[:, 1])

    v12 = inner(g1, g2)
    v23 = inner(g2, g3)
    v31 = inner(g3, g1)

    v11 = -(v12 + v31)
    v22 = -(v23 + v12)
    v33 = -(v31 + v23)
    return v11, v22, v33, v12, v23, v31


def assemble_lap_core(n, F):
    """Return an object with .asb_full / .asb_lower mirroring the MATLAB struct."""
    F = ti(F)
    f = F.shape[0]

    II = torch.cat([F[:, 0], F[:, 1], F[:, 2], F[:, 0], F[:, 1], F[:, 2],
                    F[:, 1], F[:, 2], F[:, 0]])
    JJ = torch.cat([F[:, 0], F[:, 1], F[:, 2], F[:, 1], F[:, 2], F[:, 0],
                    F[:, 0], F[:, 1], F[:, 2]])
    full = _Assembler(n, II, JJ, None, is_full=True)

    maxFF12 = torch.maximum(F[:, 0], F[:, 1])
    minFF12 = torch.minimum(F[:, 0], F[:, 1])
    maxFF23 = torch.maximum(F[:, 1], F[:, 2])
    minFF23 = torch.minimum(F[:, 1], F[:, 2])
    maxFF31 = torch.maximum(F[:, 2], F[:, 0])
    minFF31 = torch.minimum(F[:, 2], F[:, 0])

    II_l = torch.cat([F[:, 0], F[:, 1], F[:, 2], maxFF12, maxFF23, maxFF31])
    JJ_l = torch.cat([F[:, 0], F[:, 1], F[:, 2], minFF12, minFF23, minFF31])
    lower = _Assembler(n, II_l, JJ_l, None, is_full=False)

    class RL:
        pass

    out = RL()
    out.full = full
    out.lower = lower

    def asb_full(GIS, a11, a12, a22):
        return full.values(*_inner_prod_values(GIS, a11, a12, a22))

    def asb_lower(GIS, a11, a12, a22):
        return lower.values(*_inner_prod_values(GIS, a11, a12, a22))

    out.asb_full = asb_full
    out.asb_lower = asb_lower
    return out
