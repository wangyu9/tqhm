"""res_sym_grad_direct.m

A MATLAB *script* that defines a batch of elementwise anonymous functions in the
caller's workspace; here a function returning them in a dict, following
res_sym_grad_complex_plane_det1.py.

The forward map is the identity, so s_dAdP is the identity matrix; the inverse
map is the 2x2 matrix inverse, which is what the t_dAdP block differentiates.
"""

import torch


def res_sym_grad_direct():
    sym = {}

    zero = lambda p1, p2, p3: torch.zeros_like(p1)
    one = lambda p1, p2, p3: torch.ones_like(p1)

    sym['mA1'] = lambda p1, p2, p3: p1
    sym['mA2'] = lambda p1, p2, p3: p2
    sym['mA3'] = lambda p1, p2, p3: p3

    sym['np'] = 3

    sym['s_dAdP1'] = [one, zero, zero]
    sym['s_dAdP2'] = [zero, one, zero]
    sym['s_dAdP3'] = [zero, zero, one]

    D = lambda p1, p2, p3: p1 * p3 - p2 ** 2

    sym['nA1'] = lambda p1, p2, p3: p3 / D(p1, p2, p3)
    sym['nA2'] = lambda p1, p2, p3: -p2 / D(p1, p2, p3)
    sym['nA3'] = lambda p1, p2, p3: p1 / D(p1, p2, p3)

    t_dAdP11 = lambda p1, p2, p3: -p3 ** 2 / D(p1, p2, p3) ** 2
    t_dAdP12 = lambda p1, p2, p3: p2 * p3 / D(p1, p2, p3) ** 2 * 2.0
    t_dAdP13 = lambda p1, p2, p3: 1.0 / D(p1, p2, p3) - p1 * p3 / D(p1, p2, p3) ** 2

    t_dAdP21 = lambda p1, p2, p3: p2 * p3 / D(p1, p2, p3) ** 2
    t_dAdP22 = lambda p1, p2, p3: -1.0 / D(p1, p2, p3) - p2 ** 2 / D(p1, p2, p3) ** 2 * 2.0
    t_dAdP23 = lambda p1, p2, p3: p1 * p2 / D(p1, p2, p3) ** 2

    t_dAdP31 = lambda p1, p2, p3: 1.0 / D(p1, p2, p3) - p1 * p3 / D(p1, p2, p3) ** 2
    t_dAdP32 = lambda p1, p2, p3: p1 * p2 / D(p1, p2, p3) ** 2 * 2.0
    t_dAdP33 = lambda p1, p2, p3: -p1 ** 2 / D(p1, p2, p3) ** 2

    sym['t_dAdP1'] = [t_dAdP11, t_dAdP12, t_dAdP13]
    sym['t_dAdP2'] = [t_dAdP21, t_dAdP22, t_dAdP23]
    sym['t_dAdP3'] = [t_dAdP31, t_dAdP32, t_dAdP33]

    return sym
