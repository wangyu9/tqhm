"""Reusable sparse solver built on tdss.py.

Replaces the MATLAB SuiteSparse path (`analyze` for the symbolic factorization
plus `lchol` per iteration). tdss holds the symbolic analysis inside its
DirectSolver, so the layer is constructed once for a given sparsity pattern and
only `factorize()` runs per call -- matching the MATLAB intent of reusing the
symbolic factorization.

The complex solve A \\ (b_re + i*b_im) is done as a batch of two real solves,
since A is real. This mirrors the MATLAB code's use of a single complex solve to
halve the work versus solving for u and v separately.
"""

import numpy as np
import torch

import tdss
from tqhm_config import DEV, DT


class ReusableSPDSolver:
    def __init__(self, indptr, indices, n, batch_size=2):
        self.n = n
        self.batch_size = batch_size
        if torch.is_tensor(indptr):
            row_ptr = indptr.to(device=DEV, dtype=torch.int64)
            col_ind = indices.to(device=DEV, dtype=torch.int64)
        else:
            row_ptr = torch.from_numpy(np.asarray(indptr, dtype=np.int64)).to(DEV)
            col_ind = torch.from_numpy(np.asarray(indices, dtype=np.int64)).to(DEV)
        self.layer = tdss.BatchedAsymmetricSparseSolverLayer(
            row_ptr, col_ind, n, batch_size, rhs_columns=1,
            use_J_formulation=False,
        ).to(DEV)

    def solve_complex(self, data, b):
        """Solve A x = b for complex b, using a batch of 2 real right-hand sides."""
        assert self.batch_size == 2
        A2 = torch.stack([data, data])
        B2 = torch.stack([b.real.contiguous(), b.imag.contiguous()])
        X = self.layer(A2, B2).reshape(2, self.n)
        return torch.complex(X[0], X[1])

    def solve_real(self, data, b):
        assert self.batch_size == 2
        A2 = torch.stack([data, data])
        B2 = torch.stack([b.contiguous(), b.contiguous()])
        X = self.layer(A2, B2).reshape(2, self.n)
        return X[0]
