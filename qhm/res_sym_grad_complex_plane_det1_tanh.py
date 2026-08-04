"""res_sym_grad_complex_plane_det1_tanh.m

The MATLAB script starts with `error(['TBD'])`, so its body is dead code: the
'complex-plane-det1-tanh' para_type routes to res_sym_grad_complex_plane_det1
with a tanh ff/gg instead. The raise is preserved with the same message; the body
below is a transcription of the (unreachable) copy that follows it in the .m.
"""

import torch


def res_sym_grad_complex_plane_det1_tanh():
    raise NotImplementedError('TBD')

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
