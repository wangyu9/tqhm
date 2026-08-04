"""normalize_mesh.m -- scale the target UV so its area matches the source area."""

import numpy as np
import torch

from tqhm_config import DEV, DT, ti, td, npy
from doublearea import doublearea
from triangle_mesh import triangle_mesh


def normalize_mesh(V, F, VT):
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    VT = np.asarray(VT, dtype=np.float64)

    area = np.sum(0.5 * doublearea(V, F))

    mesh = triangle_mesh(V, F)

    n = V.shape[0]

    known = np.asarray(mesh['B']).ravel()

    D = mesh['D']   # torch CSR (n x n)
    BC = VT[known, :]

    # NVT = D @ (R @ [BC1, -BC0]); R scatters boundary rows into an n-vector.
    rhs = torch.zeros(n, 2, dtype=DT, device=DEV)
    rhs[ti(known)] = torch.stack([td(BC[:, 1]), td(-BC[:, 0])], dim=1)
    NVT = torch.sparse.mm(D, rhs)          # (n, 2)

    BN = npy(NVT[ti(known)])               # R.T @ NVT

    area_t = 0.5 * (BC[:, 0] @ BN[:, 0] + BC[:, 1] @ BN[:, 1])

    assert area_t > 0

    return np.sqrt(area / area_t) * VT
