"""intrinsic_grad.m -- intrinsic gradient operator built from edge lengths.

Each triangle is laid out in its own 2D frame: F[:,0] at the origin, F[:,1] at
(e3,0), F[:,2] at (..., 2*Area/e3). G is [Gx;Gy] stacked, shape (2f, n).
"""

import torch

from tqhm_config import DEV, DT, ti, td
from edge_lengths import edge_lengths
from doublearea import doublearea


def _core2d(V1, V2, V3):
    v20 = V3 - V1
    v10 = V2 - V1
    dot12 = torch.sum(v20 * v10, dim=1)
    c21 = v20[:, 0] * v10[:, 1] - v20[:, 1] * v10[:, 0]
    ns12 = c21 ** 2
    return (
        v10 * (torch.sum(v20 * v20, dim=1) / ns12)[:, None]
        + v20 * (torch.sum(v10 * v10, dim=1) / ns12)[:, None]
        - (v10 + v20) * (dot12 / ns12)[:, None]
    )


def _assemble(g1, g2, g3, F, n, f):
    """Stacked [Gx; Gy] (2f x n) CSR wrapped for row-block slicing."""
    F = ti(F)
    rows = torch.arange(f, dtype=torch.int64, device=DEV)
    # Gx occupies rows [0, f), Gy occupies rows [f, 2f); both are (f, n) blocks.
    row_idx = torch.cat([rows, rows, rows,
                         rows + f, rows + f, rows + f])
    col_idx = torch.cat([F[:, 0], F[:, 1], F[:, 2],
                         F[:, 0], F[:, 1], F[:, 2]])
    val = torch.cat([g1[:, 0], g2[:, 0], g3[:, 0],
                     g1[:, 1], g2[:, 1], g3[:, 1]])
    coo = torch.sparse_coo_tensor(torch.stack([row_idx, col_idx]), val,
                                  size=(2 * f, n)).coalesce()
    from sparse_torch import RowBlockCSR
    return RowBlockCSR(coo.to_sparse_csr())


def intrinsic_grad(V, F):
    from tqhm_config import npy
    V_np = npy(V)
    F_np = npy(F)
    F = ti(F)
    E = td(edge_lengths(V_np, F_np))
    Area = td(doublearea(V_np, F_np)) / 2.0

    n, f = int(V_np.shape[0]), int(F.shape[0])

    zeros = torch.zeros(f, dtype=DT, device=DEV)
    V1 = torch.stack([zeros, zeros], dim=1)
    V2 = torch.stack([E[:, 2], zeros], dim=1)
    V3 = torch.stack([
        (E[:, 1] ** 2 + E[:, 2] ** 2 - E[:, 0] ** 2) / (2 * E[:, 2]),
        2 * Area / E[:, 2],
    ], dim=1)

    g1 = _core2d(V1, V2, V3)[:, :2]
    g2 = _core2d(V2, V3, V1)[:, :2]
    g3 = _core2d(V3, V1, V2)[:, :2]

    G = _assemble(g1, g2, g3, F, n, f)
    return G, {'g1': g1, 'g2': g2, 'g3': g3}
