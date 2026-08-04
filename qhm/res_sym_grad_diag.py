"""res_sym_grad_diag.m

A MATLAB *script* that defines a batch of elementwise anonymous functions in the
caller's workspace; here a function returning them in a dict, following
res_sym_grad_complex_plane_det1.py.

The MATLAB closures return bare scalars (`0.0`, `1.0`) and rely on broadcasting;
they are expanded with zeros_like/ones_like so `torch.cat` in tensor_para sees
full-length columns.
"""

import torch


def res_sym_grad_diag():
    sym = {}

    zero = lambda p1, p2, p3: torch.zeros_like(p1)
    one = lambda p1, p2, p3: torch.ones_like(p1)

    sym['mA1'] = lambda p1, p2, p3: p1
    sym['mA2'] = zero
    sym['mA3'] = lambda p1, p2, p3: p3

    sym['np'] = 3

    sym['s_dAdP1'] = [one, zero, zero]
    sym['s_dAdP2'] = [zero, zero, zero]
    sym['s_dAdP3'] = [zero, zero, one]

    # t
    sym['nA1'] = lambda p1, p2, p3: 1.0 / p1
    sym['nA2'] = zero
    sym['nA3'] = lambda p1, p2, p3: 1.0 / p3

    t_dAdP11 = lambda p1, p2, p3: -1.0 / p1 ** 2
    t_dAdP33 = lambda p1, p2, p3: -1.0 / p3 ** 2

    sym['t_dAdP1'] = [t_dAdP11, zero, zero]
    sym['t_dAdP2'] = [zero, zero, zero]
    sym['t_dAdP3'] = [zero, zero, t_dAdP33]

    return sym
