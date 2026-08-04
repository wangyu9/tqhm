"""res_sym_grad_both_inverse.m

A MATLAB *script* that defines a batch of elementwise anonymous functions in the
caller's workspace; here a function returning them in a dict, following
res_sym_grad_complex_plane_det1.py.

The forward and inverse maps are the same expression here (hence the name), so
nA* == mA* and t_dAdP* == s_dAdP*.
"""

import torch


def res_sym_grad_both_inverse():
    sym = {}

    D = lambda p1, p2, p3: p1 * p3 - p2 ** 2

    sym['mA1'] = lambda p1, p2, p3: p3 / D(p1, p2, p3)
    sym['mA2'] = lambda p1, p2, p3: -p2 / D(p1, p2, p3)
    sym['mA3'] = lambda p1, p2, p3: p1 / D(p1, p2, p3)

    s_dAdP11 = lambda p1, p2, p3: -p3 ** 2 / D(p1, p2, p3) ** 2
    s_dAdP12 = lambda p1, p2, p3: p2 * p3 / D(p1, p2, p3) ** 2 * 2.0
    s_dAdP13 = lambda p1, p2, p3: 1.0 / D(p1, p2, p3) - p1 * p3 / D(p1, p2, p3) ** 2

    s_dAdP21 = lambda p1, p2, p3: p2 * p3 / D(p1, p2, p3) ** 2
    s_dAdP22 = lambda p1, p2, p3: -1.0 / D(p1, p2, p3) - p2 ** 2 / D(p1, p2, p3) ** 2 * 2.0
    s_dAdP23 = lambda p1, p2, p3: p1 * p2 / D(p1, p2, p3) ** 2

    s_dAdP31 = lambda p1, p2, p3: 1.0 / D(p1, p2, p3) - p1 * p3 / D(p1, p2, p3) ** 2
    s_dAdP32 = lambda p1, p2, p3: p1 * p2 / D(p1, p2, p3) ** 2 * 2.0
    s_dAdP33 = lambda p1, p2, p3: -p1 ** 2 / D(p1, p2, p3) ** 2

    sym['np'] = 3

    sym['s_dAdP1'] = [s_dAdP11, s_dAdP12, s_dAdP13]
    sym['s_dAdP2'] = [s_dAdP21, s_dAdP22, s_dAdP23]
    sym['s_dAdP3'] = [s_dAdP31, s_dAdP32, s_dAdP33]

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
