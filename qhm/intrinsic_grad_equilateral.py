"""intrinsic_grad_equilateral.m -- intrinsic gradient with every triangle
treated as unit equilateral (the uniform graph-Laplacian initialization)."""

import math

import torch

from tqhm_config import DEV, DT, ti
from intrinsic_grad import _core2d, _assemble


def intrinsic_grad_equilateral(n, F):
    F = ti(F)
    f = int(F.shape[0])

    zeros = torch.zeros(f, dtype=DT, device=DEV)
    ones = torch.ones(f, dtype=DT, device=DEV)
    V1 = torch.stack([zeros, zeros], dim=1)
    V2 = torch.stack([ones, zeros], dim=1)
    V3 = torch.stack([0.5 * ones, (math.sqrt(3) / 2) * ones], dim=1)

    g1 = _core2d(V1, V2, V3)[:, :2]
    g2 = _core2d(V2, V3, V1)[:, :2]
    g3 = _core2d(V3, V1, V2)[:, :2]

    G = _assemble(g1, g2, g3, F, int(n), f)
    return G, {'g1': g1, 'g2': g2, 'g3': g3}
