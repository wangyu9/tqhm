"""res_sym_grad_llt.m

A MATLAB *script* that defines a batch of elementwise anonymous functions in the
caller's workspace; here a function returning them in a dict, following
res_sym_grad_complex_plane_det1.py.

A = L*L' with L = [p1, 0; p2, p3] lower triangular.
"""

import torch


def res_sym_grad_llt():
    sym = {}

    zero = lambda p1, p2, p3: torch.zeros_like(p1)

    sym['mA1'] = lambda p1, p2, p3: p1 ** 2
    sym['mA2'] = lambda p1, p2, p3: p1 * p2
    sym['mA3'] = lambda p1, p2, p3: p2 ** 2 + p3 ** 2

    sym['np'] = 3

    s_dAdP11 = lambda p1, p2, p3: p1 * 2.0
    s_dAdP21 = lambda p1, p2, p3: p2
    s_dAdP22 = lambda p1, p2, p3: p1
    s_dAdP32 = lambda p1, p2, p3: p2 * 2.0
    s_dAdP33 = lambda p1, p2, p3: p3 * 2.0

    sym['s_dAdP1'] = [s_dAdP11, zero, zero]
    sym['s_dAdP2'] = [s_dAdP21, s_dAdP22, zero]
    sym['s_dAdP3'] = [zero, s_dAdP32, s_dAdP33]

    sym['nA1'] = lambda p1, p2, p3: 1.0 / p1 ** 2 / p3 ** 2 * (p2 ** 2 + p3 ** 2)
    sym['nA2'] = lambda p1, p2, p3: -(p2 / p3 ** 2) / p1
    sym['nA3'] = lambda p1, p2, p3: 1.0 / p3 ** 2

    t_dAdP11 = lambda p1, p2, p3: 1.0 / p1 ** 3 / p3 ** 2 * (p2 ** 2 + p3 ** 2) * -2.0
    t_dAdP12 = lambda p1, p2, p3: 1.0 / p1 ** 2 * p2 / p3 ** 2 * 2.0
    t_dAdP13 = lambda p1, p2, p3: (1.0 / p1 ** 2 * 2.0) / p3 \
        - 1.0 / p1 ** 2 / p3 ** 3 * (p2 ** 2 + p3 ** 2) * 2.0

    t_dAdP21 = lambda p1, p2, p3: 1.0 / p1 ** 2 * p2 / p3 ** 2
    t_dAdP22 = lambda p1, p2, p3: -1.0 / p3 ** 2 / p1
    t_dAdP23 = lambda p1, p2, p3: (p2 / p3 ** 3 * 2.0) / p1

    t_dAdP33 = lambda p1, p2, p3: 1.0 / p3 ** 3 * -2.0

    sym['t_dAdP1'] = [t_dAdP11, t_dAdP12, t_dAdP13]
    sym['t_dAdP2'] = [t_dAdP21, t_dAdP22, t_dAdP23]
    sym['t_dAdP3'] = [zero, zero, t_dAdP33]

    return sym
