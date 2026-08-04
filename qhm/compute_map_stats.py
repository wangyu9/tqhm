"""compute_map_stats.m -- singular values of the per-face Jacobian of the map.

MATLAB loops over faces calling svd on each 2x2 block; here the whole (f,2,2)
stack goes through torch.linalg.svdvals at once, which gives the same
descending-order singular values.
"""

import numpy as np
import torch

from tqhm_config import td


def compute_map_stats(mesh, UV):
    UV = td(UV)
    u = UV[:, 0]
    v = UV[:, 1]

    stats = {}

    V = mesh['V']
    F = mesh['F']

    G = mesh['G']

    n = V.shape[0]
    f = F.shape[0]

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
    return stats


def _matvec(A, x):
    """A is scipy sparse (one-time setup stays in scipy) and x is a tensor."""
    if torch.is_tensor(A):
        return A @ x
    y = A @ x.detach().cpu().numpy()
    return torch.as_tensor(np.asarray(y), dtype=x.dtype, device=x.device)
