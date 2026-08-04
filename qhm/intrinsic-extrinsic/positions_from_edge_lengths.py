"""positions_from_edge_lengths.m -- lay each triangle out in the plane from its
three edge lengths, vertex 1 at the origin and vertex 2 on the +x axis.

Returns V as f x 3 x 2 (triangle, vertex, coordinate).
"""

import torch


def positions_from_edge_lengths(E):
    # positions_from_edge_lengths(edge_lengths([[0,0],[1.21,0],[0.3,0.37]], [[0,1,2]]))

    f = E.shape[0]

    V = torch.zeros(f, 3, 2, dtype=E.dtype, device=E.device)

    V[:, 1, 0] = E[:, 2]

    S = (E[:, 0] + E[:, 1] + E[:, 2]) / 2

    # https://en.wikipedia.org/wiki/Heron%27s_formula
    V[:, 2, 0] = (E[:, 1] ** 2 + E[:, 2] ** 2 - E[:, 0] ** 2) / (2 * E[:, 2])
    V[:, 2, 1] = 2 * torch.sqrt(S * (S - E[:, 0]) * (S - E[:, 1]) * (S - E[:, 2])) / E[:, 2]

    return V
