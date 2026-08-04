"""tutte_init.m -- harmonic (Tutte) interior positions for a prescribed boundary.

Solves L(UB,UB) VT(UB) = -L(UB,B) VT(B) with L = GI' MF GI, keeping VTI's boundary
values. The two right-hand sides (x and y) are packed into one complex solve, so
the single tdss ReusableSPDSolver call does both at once.
"""

import numpy as np
import scipy.sparse as sp
import torch

from tqhm_config import td, npy
from doublearea import doublearea
from intrinsic_grad import intrinsic_grad
from intrinsic_grad_equilateral import intrinsic_grad_equilateral
from circular_laplacian import circular_laplacian
from tdss_solver import ReusableSPDSolver


def tutte_init(V, F, VTI, graph):
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    VTI = np.asarray(VTI, dtype=np.float64)

    n = V.shape[0]
    f = F.shape[0]

    if graph:
        print('Init with uniform graph Laplacian')
        GI, _ = intrinsic_grad_equilateral(n, F)
        Area = np.ones(F.shape[0])
    else:
        GI, _ = intrinsic_grad(V, F)
        Area = doublearea(V, F) / 2

    MF = sp.diags(np.concatenate([Area, Area]), 0, shape=(2 * f, 2 * f), format='csr')

    L = (GI.T @ MF @ GI).tocsr()

    BE, _, _ = circular_laplacian(V.shape[0], F)
    B = np.sort(BE[:, 0])

    mask = np.zeros(n, dtype=bool)
    mask[B] = True
    UB = np.flatnonzero(~mask)

    VT = VTI[:, 0:2].copy()

    Luu = L[UB, :][:, UB].tocsr()
    Lub = L[UB, :][:, B].tocsr()

    rhs = -(Lub @ VT[B, :])

    solver = ReusableSPDSolver(Luu.indptr, Luu.indices, UB.size, batch_size=2)
    x = solver.solve_complex(td(Luu.data), torch.complex(td(rhs[:, 0]), td(rhs[:, 1])))

    VT[UB, 0] = npy(x.real)
    VT[UB, 1] = npy(x.imag)
    return VT
