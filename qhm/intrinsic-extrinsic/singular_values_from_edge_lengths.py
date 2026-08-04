"""singular_values_from_edge_lengths.m -- squared singular values of the affine map
taking each rest triangle (edge lengths E0) to its deformed counterpart (E).

Returns [A + 2*sqrt(ysq), A - 2*sqrt(ysq)], i.e. sigma1^2 and sigma2^2; the
`intrinsic_grad_hessian` symbolic code takes the square root of these.
"""

import torch


def singular_values_from_edge_lengths(E, E0):
    a0 = E0[:, 0] ** 2
    b0 = E0[:, 1] ** 2
    c0 = E0[:, 2] ** 2

    a1 = E[:, 0] ** 2
    b1 = E[:, 1] ** 2
    c1 = E[:, 2] ** 2

    R = torch.sqrt(2 * a0 * b0 + 2 * a0 * c0 + 2 * b0 * c0 - a0 ** 2 - b0 ** 2 - c0 ** 2)

    A = ((-a0 + b0 + c0) * a1 + (a0 - b0 + c0) * b1 + (a0 + b0 - c0) * c1) / R ** 2

    ysq = ((b0 * c0) * a1 ** 2 + (a0 * c0) * b1 ** 2 + (a0 * b0) * c1 ** 2
           + (-a0 - b0 + c0) * c0 * b1 * a1
           + (-a0 + b0 - c0) * b0 * a1 * c1
           + (a0 - b0 - c0) * a0 * c1 * b1
           ) / R ** 4

    return torch.stack([A + 2 * torch.sqrt(ysq), A - 2 * torch.sqrt(ysq)], dim=1)
