"""triangle_mesh_basic.m -- the light-weight mesh struct used by the solver."""

import numpy as np
import torch

from tqhm_config import td, ti
from doublearea import doublearea
from outline import outline
from mesh_dirac import mesh_dirac
from intrinsic_grad import intrinsic_grad
from sparse_torch import SpOp


def _basis_grad_core(V, F):
    if V.shape[1] == 2:
        V = np.c_[V, np.zeros(V.shape[0])]
    v20 = V[F[:, 2], :] - V[F[:, 0], :]
    v10 = V[F[:, 1], :] - V[F[:, 0], :]
    dot12 = np.sum(v20 * v10, axis=1)
    ns12 = np.sum(np.cross(v20, v10) ** 2, axis=1)
    return (
        v10 * (np.sum(v20 * v20, axis=1) / ns12)[:, None]
        + v20 * (np.sum(v10 * v10, axis=1) / ns12)[:, None]
        - (v10 + v20) * (dot12 / ns12)[:, None]
    )


def triangle_mesh_basic(V, F, indEC=None):
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)

    n = V.shape[0]
    f = F.shape[0]

    dim = F.shape[1] - 1
    assert dim in (2, 3)

    mesh = {'n': n, 'f': f, 'dim': dim}
    mesh['V_np'] = V
    mesh['F_np'] = F
    mesh['V'] = td(V)
    mesh['F'] = ti(F)

    mesh['D'] = mesh_dirac(n, F)
    mesh['FA'] = doublearea(V, F) / 2.0

    BE = outline(F)
    B = BE[:, 0]
    mesh['BE'] = BE

    if indEC is None or len(np.atleast_1d(indEC)) == 0:
        IKB = B
    else:
        IKB = np.concatenate([B, np.asarray(indEC, dtype=np.int64).ravel()])

    mask = np.zeros(n, dtype=bool)
    mask[IKB] = True
    IUB = np.flatnonzero(~mask)

    assert IKB.size == np.unique(IKB).size

    mesh['IKB_np'] = IKB
    mesh['IUB_np'] = IUB
    mesh['IKB'] = ti(IKB)
    mesh['IUB'] = ti(IUB)

    g1 = _basis_grad_core(V, F[:, [0, 1, 2]])[:, :2]
    g2 = _basis_grad_core(V, F[:, [1, 2, 0]])[:, :2]
    g3 = _basis_grad_core(V, F[:, [2, 0, 1]])[:, :2]
    mesh['g1'], mesh['g2'], mesh['g3'] = g1, g2, g3

    GI, GIS = intrinsic_grad(V, F)
    mesh['GI_sp'] = GI
    mesh['GI'] = SpOp(GI)
    mesh['GIS'] = {k: td(v) for k, v in GIS.items()}
    mesh['GIS_np'] = GIS

    return mesh
