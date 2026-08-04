"""mesh_dirac.m"""

import torch

from tqhm_config import DEV, DT, ti


def mesh_dirac(n, F):
    F = ti(F)
    f = F.shape[0]
    ones = torch.ones(3 * f, dtype=DT, device=DEV)
    rows = torch.cat([F[:, 0], F[:, 1], F[:, 2]])
    cols = torch.cat([F[:, 1], F[:, 2], F[:, 0]])
    idx = torch.stack([rows, cols])
    Dh = torch.sparse_coo_tensor(idx, ones, size=(n, n)).coalesce()
    D = (Dh - Dh.transpose(0, 1)).coalesce()
    return 0.5 * D.to_sparse_csr()
