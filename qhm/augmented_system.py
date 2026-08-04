"""augmented_system.m -- KKT/saddle-point solve for the constrained step.

The block matrix has zero diagonal blocks (`ZZA`, `ZRR`) and mixes `+LLB` with
`-LLA`, so it is symmetric *indefinite*, never SPD. Cholesky (and hence
tdss_solver.ReusableSPDSolver) is therefore invalid here: MATLAB's `lhs \\ rhs`
picks an LU/LDL path for such a matrix. This port uses
`scipy.sparse.linalg.splu` (an LU factorization from SuperLU, not
SuiteSparse/CHOLMOD) on the assembled sparse system.

The sparsity pattern depends on the caller's `RU`/`RB`, so nothing is cached.

Deviation: MATLAB's empty-constraint defaults `RU = zeros(0,size(LA,1))` and
`RB = zeros(0,size(LB,1))` are half as wide as the blocks they sit next to
(`RU'` needs 2*nu rows and `RB'` needs 2*nk), so MATLAB's own concatenation
would reject them. They are widened to the dimensionally consistent 0-row
empties here; with r == 0 the blocks contribute nothing either way.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from tqhm_config import npy


def _sp_zeros_like(AA):
    return sp.csr_matrix((AA.shape[0], AA.shape[1]))


def _zeros_like(AA):
    return np.zeros(np.shape(AA))


def _sparse(M):
    return sp.csr_matrix(M)


def augmented_system(H, dW, pre, L, LA, LB, RU, RB, eq_rhs):
    LA = _sparse(LA)
    LB = _sparse(LB)

    empty_eq = eq_rhs is None or np.size(eq_rhs) == 0
    if empty_eq:
        # widened from MATLAB's zeros(0,size(LA,1)) / zeros(0,size(LB,1))
        RU = np.zeros((0, 2 * LA.shape[0]))
        RB = np.zeros((0, 2 * LB.shape[1]))
        eq_rhs = np.zeros((0, 1))

    RU = _sparse(np.atleast_2d(npy(RU)))
    RB = _sparse(np.atleast_2d(npy(RB)))
    eq_rhs = npy(eq_rhs).reshape(-1)

    H = _sparse(H)

    n = pre['n'] if isinstance(pre, dict) else pre.n
    dim = pre['dim'] if isinstance(pre, dict) else pre.dim
    known = npy(pre['known'] if isinstance(pre, dict) else pre.known).ravel()
    unknown = npy(pre['unknown'] if isinstance(pre, dict) else pre.unknown).ravel()

    r = eq_rhs.size

    gg = npy(dW).reshape(-1, order='F')
    assert dim == 2
    # known: 2, unknown: 1.
    id2 = np.concatenate([known, known + n])
    id1 = np.concatenate([unknown, unknown + n])

    # H = [L,0;0,L] + HC;
    H22 = H[id2, :][:, id2]
    H11 = H[id1, :][:, id1]
    H12 = H[id1, :][:, id2]
    H21 = H[id2, :][:, id1]

    LLA = sp.bmat([[LA, _sp_zeros_like(LA)], [_sp_zeros_like(LA), LA]], format='csr')
    LLB = sp.bmat([[LB, _sp_zeros_like(LB)], [_sp_zeros_like(LB), LB]], format='csr')

    g1 = gg[id1]
    g2 = gg[id2]

    ZZA = _sp_zeros_like(LLA)
    ZZB = _sp_zeros_like(LLB)
    Zg1 = _zeros_like(g1)
    ZRA = sp.csr_matrix((r, LLA.shape[0]))
    ZRR = sp.csr_matrix((r, r))

    if False:
        lhs = sp.bmat([
            [H22, H21, LLB.T, LLB.T, ZZB.T],
            [H12, ZZA, -LLA, ZZA, ZZA],
            [LLB, -LLA, ZZA, ZZA, ZZA],
            [LLB, ZZA, ZZA, ZZA, LLA],
            [ZZB, ZZA, ZZA, LLA, H11],
        ], format='csr')

        rhs = np.concatenate([g2, g1, Zg1, Zg1, Zg1])

    lhs = sp.bmat([
        [H22, H21, LLB.T, RB.T],
        [H12, H11, -LLA, RU.T],
        [LLB, -LLA, ZZA, ZRA.T],
        [RB, RU, ZRA, ZRR],
    ], format='csc')

    rhs = np.concatenate([g2, g1, Zg1, eq_rhs])

    # indefinite by construction -> LU, not Cholesky
    r = splu(lhs).solve(rhs)

    xxx = r[0:g2.size]

    return xxx
