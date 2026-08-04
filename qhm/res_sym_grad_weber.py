"""res_sym_grad_weber.m

A MATLAB *script* that defines a batch of elementwise anonymous functions in the
caller's workspace; here a function returning them in a dict, following
res_sym_grad_complex_plane_det1.py.

Note MATLAB's s_dAdP11/s_dAdP31 differentiate sqrt(p1), so p1 must be positive.
"""

import torch


def res_sym_grad_weber():
    sym = {}

    zero = lambda p1, p2, p3: torch.zeros_like(p1)
    two = lambda p1, p2, p3: torch.full_like(p1, 2.0)
    mtwo = lambda p1, p2, p3: torch.full_like(p1, -2.0)

    sym['mA1'] = lambda p1, p2, p3: p2 * 2.0 + torch.sqrt(p1)
    sym['mA2'] = lambda p1, p2, p3: p3 * 2.0
    sym['mA3'] = lambda p1, p2, p3: p2 * -2.0 + torch.sqrt(p1)

    sym['np'] = 3

    s_dAdP11 = lambda p1, p2, p3: 1.0 / torch.sqrt(p1) / 2.0
    s_dAdP31 = lambda p1, p2, p3: 1.0 / torch.sqrt(p1) / 2.0

    sym['s_dAdP1'] = [s_dAdP11, two, zero]
    sym['s_dAdP2'] = [zero, zero, two]
    sym['s_dAdP3'] = [s_dAdP31, mtwo, zero]

    D = lambda p1, p2, p3: -p1 + p2 ** 2 * 4.0 + p3 ** 2 * 4.0

    sym['nA1'] = lambda p1, p2, p3: (p2 * 2.0 - torch.sqrt(p1)) / D(p1, p2, p3)
    sym['nA2'] = lambda p1, p2, p3: (p3 * 2.0) / D(p1, p2, p3)
    sym['nA3'] = lambda p1, p2, p3: -(p2 * 2.0 + torch.sqrt(p1)) / D(p1, p2, p3)

    t_dAdP11 = lambda p1, p2, p3: (1.0 / torch.sqrt(p1) * (-1.0 / 2.0)) / D(p1, p2, p3) \
        + (p2 * 2.0 - torch.sqrt(p1)) / D(p1, p2, p3) ** 2
    t_dAdP12 = lambda p1, p2, p3: 2.0 / D(p1, p2, p3) \
        - p2 * (p2 * 2.0 - torch.sqrt(p1)) / D(p1, p2, p3) ** 2 * 8.0
    t_dAdP13 = lambda p1, p2, p3: p3 * (p2 * 2.0 - torch.sqrt(p1)) / D(p1, p2, p3) ** 2 * -8.0

    t_dAdP21 = lambda p1, p2, p3: p3 / D(p1, p2, p3) ** 2 * 2.0
    t_dAdP22 = lambda p1, p2, p3: p2 * p3 / D(p1, p2, p3) ** 2 * -1.6e+1
    t_dAdP23 = lambda p1, p2, p3: p3 ** 2 / D(p1, p2, p3) ** 2 * -1.6e+1 + 2.0 / D(p1, p2, p3)

    t_dAdP31 = lambda p1, p2, p3: (1.0 / torch.sqrt(p1) * (-1.0 / 2.0)) / D(p1, p2, p3) \
        - (p2 * 2.0 + torch.sqrt(p1)) / D(p1, p2, p3) ** 2
    t_dAdP32 = lambda p1, p2, p3: -2.0 / D(p1, p2, p3) \
        + p2 * (p2 * 2.0 + torch.sqrt(p1)) / D(p1, p2, p3) ** 2 * 8.0
    t_dAdP33 = lambda p1, p2, p3: p3 * (p2 * 2.0 + torch.sqrt(p1)) / D(p1, p2, p3) ** 2 * 8.0

    sym['t_dAdP1'] = [t_dAdP11, t_dAdP12, t_dAdP13]
    sym['t_dAdP2'] = [t_dAdP21, t_dAdP22, t_dAdP23]
    sym['t_dAdP3'] = [t_dAdP31, t_dAdP32, t_dAdP33]

    return sym
