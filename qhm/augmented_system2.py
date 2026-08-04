"""augmented_system2.m -- the alternative saddle-point system.

Same story as augmented_system.py: the middle diagonal block is exactly zero
(`Zu2u2`) and the last one is a tiny *positive* 1e-6*I regularizer, so the
overall matrix is symmetric indefinite. Solved with
`scipy.sparse.linalg.splu` (SuperLU LU, not SuiteSparse/CHOLMOD); Cholesky-based
ReusableSPDSolver would be invalid.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from tqhm_config import npy


def _sp_zeros_like(AA):
    return sp.csr_matrix((AA.shape[0], AA.shape[1]))


def augmented_system2(H, ga, pre, L, extra, RU, RB, eq_rhs):
    if eq_rhs is None or np.size(eq_rhs) == 0:
        eq_rhs = np.zeros((0, 1))

    ga = npy(ga).reshape(-1)

    n = pre['n'] if isinstance(pre, dict) else pre.n
    dim = pre['dim'] if isinstance(pre, dict) else pre.dim
    known = npy(pre['known'] if isinstance(pre, dict) else pre.known).ravel()
    unknown = npy(pre['unknown'] if isinstance(pre, dict) else pre.unknown).ravel()

    nu = unknown.size
    na = ga.size

    r = npy(eq_rhs).reshape(-1).size

    assert dim == 2
    # known: 2, unknown: 1.
    id2 = np.concatenate([known, known + n])
    id1 = np.concatenate([unknown, unknown + n])

    H = sp.csr_matrix(H)
    L = sp.csr_matrix(L)

    # H = [L,0;0,L] + HC;
    LAB = L[unknown, :]
    LAB2 = sp.bmat([[LAB, _sp_zeros_like(LAB)], [_sp_zeros_like(LAB), LAB]],
                   format='csr')

    Oaa = sp.eye(na, format='csr')
    Zan2 = sp.csr_matrix((na, n * 2))
    Zu2u2 = sp.csr_matrix((nu * 2, nu * 2))

    Zn2 = np.zeros(n * 2)
    Zu2 = np.zeros(nu * 2)

    assert dim == 2
    Tx = sp.csr_matrix(extra['Tx'] if isinstance(extra, dict) else extra.Tx)
    Ty = sp.csr_matrix(extra['Ty'] if isinstance(extra, dict) else extra.Ty)
    T = sp.hstack([Tx, Ty], format='csr')

    lhs = sp.bmat([
        [H, -LAB2.T, Zan2.T],
        [-LAB2, Zu2u2, T.T],
        [Zan2, T, 1e-6 * Oaa],
    ], format='csc')

    rhs = np.concatenate([Zn2, Zu2, ga])

    # indefinite by construction -> LU, not Cholesky
    r = splu(lhs).solve(rhs)

    xxx = r[r.size - na:]

    return xxx
