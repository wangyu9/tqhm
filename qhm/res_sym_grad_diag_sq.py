"""res_sym_grad_diag_sq.m

A MATLAB *script* that defines a batch of elementwise anonymous functions in the
caller's workspace; here a function returning them in a dict, following
res_sym_grad_complex_plane_det1.py.
"""

import torch


def res_sym_grad_diag_sq():
    sym = {}

    zero = lambda p1, p2, p3: torch.zeros_like(p1)

    sym['mA1'] = lambda p1, p2, p3: p1 ** 2
    sym['mA2'] = zero
    sym['mA3'] = lambda p1, p2, p3: p3 ** 2

    sym['np'] = 3

    s_dAdP11 = lambda p1, p2, p3: p1 * 2.0
    s_dAdP33 = lambda p1, p2, p3: p3 * 2.0

    sym['s_dAdP1'] = [s_dAdP11, zero, zero]
    sym['s_dAdP2'] = [zero, zero, zero]
    sym['s_dAdP3'] = [zero, zero, s_dAdP33]

    sym['nA1'] = lambda p1, p2, p3: 1.0 / p1 ** 2
    sym['nA2'] = zero
    sym['nA3'] = lambda p1, p2, p3: 1.0 / p3 ** 2

    t_dAdP11 = lambda p1, p2, p3: 1.0 / p1 ** 3 * -2.0
    t_dAdP33 = lambda p1, p2, p3: 1.0 / p3 ** 3 * -2.0

    sym['t_dAdP1'] = [t_dAdP11, zero, zero]
    sym['t_dAdP2'] = [zero, zero, zero]
    sym['t_dAdP3'] = [zero, zero, t_dAdP33]

    return sym
