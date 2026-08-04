"""torch sparse helpers (no MATLAB counterpart).

torch's sparse CSR matmul requires matching dtypes, so a real operator cannot be
applied directly to a complex vector. SpOp therefore stores the explicit
transpose and applies a complex vector as one real spmm over a 2-column
[real, imag] block, which is also faster than two separate matvecs.

`csr_row_block` / `RowBlockCSR` exist because torch CSR tensors cannot be sliced
natively (`A[:f, :]` raises "Sparse CSR tensors do not have strides"). A
contiguous row block is a trivial slice of the crow/col/val arrays, so the
gradient operator can be handed around as one stacked CSR and sliced into its
Gx / Gy halves on demand.
"""

import torch

from tqhm_config import DEV, DT


def _to_torch_csr(M):
    """Accept a torch CSR (returned as-is on DEV/DT) or a scipy matrix."""
    if torch.is_tensor(M):
        if M.layout is torch.sparse_csr:
            return M.to(device=DEV, dtype=DT)
        return M.to(device=DEV, dtype=DT).to_sparse_csr()
    if isinstance(M, RowBlockCSR):
        return M.csr
    import numpy as np
    import scipy.sparse as sp
    M = sp.csr_matrix(M)
    return torch.sparse_csr_tensor(
        torch.from_numpy(M.indptr.astype(np.int64)),
        torch.from_numpy(M.indices.astype(np.int64)),
        torch.from_numpy(M.data.astype(np.float64)),
        size=M.shape, dtype=DT, device=DEV,
    )


def scipy_to_torch(M):
    """Kept for callers that still pass a scipy matrix; delegates to _to_torch_csr."""
    return _to_torch_csr(M)


def csr_row_block(A, a, b):
    """Rows [a, b) of a torch CSR tensor as a new torch CSR tensor."""
    crow, col, val = A.crow_indices(), A.col_indices(), A.values()
    s, e = int(crow[a]), int(crow[b])
    return torch.sparse_csr_tensor(crow[a:b + 1] - crow[a], col[s:e], val[s:e],
                                   size=(b - a, A.shape[1]))


class RowBlockCSR:
    """A torch CSR matrix that supports contiguous row-block slicing.

    Only the two slice forms the solver uses are supported:
    ``M[:k, :]`` and ``M[k:m, :]`` (a full-column contiguous row range).
    """

    def __init__(self, csr):
        self.csr = _to_torch_csr(csr)
        self.shape = tuple(self.csr.shape)

    def __getitem__(self, key):
        rows, cols = key
        assert isinstance(cols, slice) and cols == slice(None), \
            'RowBlockCSR only supports full-column row-block slices'
        a = 0 if rows.start is None else int(rows.start)
        b = self.shape[0] if rows.stop is None else int(rows.stop)
        assert rows.step in (None, 1)
        return csr_row_block(self.csr, a, b)

    def to_torch_csr(self):
        return self.csr


class SpOp:
    """A real sparse operator that can be applied to real or complex vectors."""

    def __init__(self, M):
        A = _to_torch_csr(M)
        self.shape = tuple(A.shape)
        self.A = A
        self.AT = A.t().to_sparse_csr()

    def _apply(self, A, x):
        if x.is_complex():
            rhs = torch.stack([x.real, x.imag], dim=1).contiguous()
            y = A @ rhs
            return torch.complex(y[:, 0], y[:, 1])
        return A @ x

    def matvec(self, x):
        return self._apply(self.A, x)

    def rmatvec(self, x):
        return self._apply(self.AT, x)

    def __matmul__(self, x):
        return self.matvec(x)
