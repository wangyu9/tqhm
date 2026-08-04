"""edge_length_per_tri.m -- per-triangle edge lengths, opposite-vertex ordering.

The MATLAB source indexes `V(F(:,2))` with a single subscript, which on an n x 2
array linearizes column-major and therefore only reads the first column for
n >= max(F). This port keeps the intended row indexing `V[F[:,1], :]`, which is
what the only caller (EnergyClassSymDirichlet.ProxyQuasi) wants.
"""

import torch


def _rnr(x):
    return torch.sqrt(torch.sum(x ** 2, dim=1))


def edge_length_per_tri(V, F):
    return torch.stack([
        _rnr(V[F[:, 1], :] - V[F[:, 2], :]),
        _rnr(V[F[:, 2], :] - V[F[:, 0], :]),
        _rnr(V[F[:, 0], :] - V[F[:, 1], :]),
    ], dim=1)
