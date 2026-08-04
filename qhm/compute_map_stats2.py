"""compute_map_stats2.m -- compute_map_stats taking the gradient operator
directly, plus the MIPS distortion."""

import torch

from tqhm_config import td
from compute_map_stats import _matvec


def compute_map_stats2(G, UV):
    UV = td(UV)
    u = UV[:, 0]
    v = UV[:, 1]

    stats = {}

    # n = UV.shape[0]
    f = G.shape[0] // 2

    Gx = G[:f, :]
    Gy = G[f:2 * f, :]

    Js = torch.zeros(f, 2, 2, dtype=UV.dtype, device=UV.device)

    Js[:, 0, 0] = _matvec(Gx, u)
    Js[:, 0, 1] = _matvec(Gy, u)

    Js[:, 1, 0] = _matvec(Gx, v)
    Js[:, 1, 1] = _matvec(Gy, v)

    ss = torch.linalg.svdvals(Js)

    stats['sigmas'] = ss
    stats['sigma_ratio'] = ss[:, 0] / ss[:, 1]

    stats['MIPS'] = stats['sigma_ratio'] + 1.0 / stats['sigma_ratio']
    return stats
