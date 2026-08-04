"""res_sym_grad_complex_sym_det1.m

A MATLAB *script* that defines a batch of elementwise anonymous functions in the
caller's workspace; here a function returning them in a dict, following
res_sym_grad_complex_plane_det1.py.

Same expressions as the inner (pre-squashing) part of
res_sym_grad_complex_plane_det1, but the variable itself is the point in the
unit disc, so there is no ff/gg chain rule.
"""

import torch


def res_sym_grad_complex_sym_det1():
    sym = {}

    zero = lambda p1, p2, p3: torch.zeros_like(p1)

    D = lambda p1, p2, p3: p1 ** 2 + p2 ** 2 - 1.0

    sym['mA1'] = lambda p1, p2, p3: -((p1 - 1.0) ** 2 + p2 ** 2) / D(p1, p2, p3)
    sym['mA2'] = lambda p1, p2, p3: (p2 * 2.0) / D(p1, p2, p3)
    sym['mA3'] = lambda p1, p2, p3: -((p1 + 1.0) ** 2 + p2 ** 2) / D(p1, p2, p3)

    sym['np'] = 3

    s_dAdP11 = lambda p1, p2, p3: -(p1 * 2.0 - 2.0) / D(p1, p2, p3) \
        + p1 * ((p1 - 1.0) ** 2 + p2 ** 2) / D(p1, p2, p3) ** 2 * 2.0
    s_dAdP12 = lambda p1, p2, p3: (p2 * -2.0) / D(p1, p2, p3) \
        + p2 * ((p1 - 1.0) ** 2 + p2 ** 2) / D(p1, p2, p3) ** 2 * 2.0

    s_dAdP21 = lambda p1, p2, p3: p1 * p2 / D(p1, p2, p3) ** 2 * -4.0
    s_dAdP22 = lambda p1, p2, p3: 2.0 / D(p1, p2, p3) \
        - p2 ** 2 / D(p1, p2, p3) ** 2 * 4.0

    s_dAdP31 = lambda p1, p2, p3: -(p1 * 2.0 + 2.0) / D(p1, p2, p3) \
        + p1 * ((p1 + 1.0) ** 2 + p2 ** 2) / D(p1, p2, p3) ** 2 * 2.0
    s_dAdP32 = lambda p1, p2, p3: (p2 * -2.0) / D(p1, p2, p3) \
        + p2 * ((p1 + 1.0) ** 2 + p2 ** 2) / D(p1, p2, p3) ** 2 * 2.0

    sym['s_dAdP1'] = [s_dAdP11, s_dAdP12, zero]
    sym['s_dAdP2'] = [s_dAdP21, s_dAdP22, zero]
    sym['s_dAdP3'] = [s_dAdP31, s_dAdP32, zero]

    sym['nA1'] = lambda p1, p2, p3: -((p1 + 1.0) ** 2 + p2 ** 2) / D(p1, p2, p3)
    sym['nA2'] = lambda p1, p2, p3: (p2 * -2.0) / D(p1, p2, p3)
    sym['nA3'] = lambda p1, p2, p3: -((p1 - 1.0) ** 2 + p2 ** 2) / D(p1, p2, p3)

    t_dAdP11 = lambda p1, p2, p3: -(p1 * 2.0 + 2.0) / D(p1, p2, p3) \
        + p1 * ((p1 + 1.0) ** 2 + p2 ** 2) / D(p1, p2, p3) ** 2 * 2.0
    t_dAdP12 = lambda p1, p2, p3: (p2 * -2.0) / D(p1, p2, p3) \
        + p2 * ((p1 + 1.0) ** 2 + p2 ** 2) / D(p1, p2, p3) ** 2 * 2.0

    t_dAdP21 = lambda p1, p2, p3: p1 * p2 / D(p1, p2, p3) ** 2 * 4.0
    t_dAdP22 = lambda p1, p2, p3: -2.0 / D(p1, p2, p3) \
        + p2 ** 2 / D(p1, p2, p3) ** 2 * 4.0

    t_dAdP31 = lambda p1, p2, p3: -(p1 * 2.0 - 2.0) / D(p1, p2, p3) \
        + p1 * ((p1 - 1.0) ** 2 + p2 ** 2) / D(p1, p2, p3) ** 2 * 2.0
    t_dAdP32 = lambda p1, p2, p3: (p2 * -2.0) / D(p1, p2, p3) \
        + p2 * ((p1 - 1.0) ** 2 + p2 ** 2) / D(p1, p2, p3) ** 2 * 2.0

    sym['t_dAdP1'] = [t_dAdP11, t_dAdP12, zero]
    sym['t_dAdP2'] = [t_dAdP21, t_dAdP22, zero]
    sym['t_dAdP3'] = [t_dAdP31, t_dAdP32, zero]

    return sym
