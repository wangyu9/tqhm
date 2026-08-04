"""compute_map_stats3.m -- the full distortion report for a map (V,F) -> UV:
singular values, MIPS, ARAP, isometric distortion and the flip count.

`wavg_MIPS` is per-face (MIPS weighted by the area fraction), not a scalar;
print_map_stats3 sums it. Field names are kept as in MATLAB, including the
`MIPS_old` name.
"""

import numpy as np
import torch

from tqhm_config import td, npy
from intrinsic_grad import intrinsic_grad
from doublearea import doublearea
from compute_map_stats import _matvec


def compute_map_stats3(V, F, UV):
    UV_t = td(UV)
    u = UV_t[:, 0]
    v = UV_t[:, 1]

    stats = {}

    G, _ = intrinsic_grad(V, F)
    Area = np.abs(doublearea(V, F)) / 2.0

    # n = UV.shape[0]
    f = G.shape[0] // 2

    Gx = G[:f, :]
    Gy = G[f:2 * f, :]

    Js = torch.zeros(f, 2, 2, dtype=UV_t.dtype, device=UV_t.device)

    Js[:, 0, 0] = _matvec(Gx, u)
    Js[:, 0, 1] = _matvec(Gy, u)

    Js[:, 1, 0] = _matvec(Gx, v)
    Js[:, 1, 1] = _matvec(Gy, v)

    ss = torch.linalg.svdvals(Js)

    stats['sigmas'] = ss
    stats['sigma_ratio'] = ss[:, 0] / ss[:, 1]

    Area_t = td(Area)

    stats['MIPS_old'] = stats['sigma_ratio'] + 1.0 / stats['sigma_ratio']

    stats['wavg_MIPS'] = (stats['sigma_ratio'] + 1.0 / stats['sigma_ratio']) \
        * Area_t / Area_t.sum()

    flipped = doublearea(np.stack([npy(u), npy(v)], axis=1), F) < 0

    stats['num_flips'] = int(np.count_nonzero(flipped))

    stats['Area'] = Area

    # stats['get_wavg_ARAP'] = lambda: ...
    # mean(sum((stats.sigmas-1)**2, axis=1))

    stats['arap'] = torch.sum((stats['sigmas'] - 1.0) ** 2, dim=1)

    # mean(max([stats.sigmas[:,0], 1/stats.sigmas[:,1]], axis=1))

    stats['iso'] = torch.maximum(stats['sigmas'][:, 0], 1.0 / stats['sigmas'][:, 1])
    return stats
