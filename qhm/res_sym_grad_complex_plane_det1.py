"""res_sym_grad_complex_plane_det1.m

A MATLAB *script* (not a function) that defines a batch of elementwise anonymous
functions in the caller's workspace. Here it is a function returning them in a
dict, since Python has no script-level workspace injection.

`ff` and `gg = ff'` are supplied by the caller (tensor_para) and select the
squashing profile: the log variant for 'complex-plane-det1', tanh for
'complex-plane-det1-tanh'.
"""

import torch


def res_sym_grad_complex_plane_det1(ff, gg):
    sym = {}

    pmA1 = lambda p1, p2, p3: -((p1 - 1.0) ** 2 + p2 ** 2) / (p1 ** 2 + p2 ** 2 - 1.0)
    pmA2 = lambda p1, p2, p3: (p2 * 2.0) / (p1 ** 2 + p2 ** 2 - 1.0)
    pmA3 = lambda p1, p2, p3: -((p1 + 1.0) ** 2 + p2 ** 2) / (p1 ** 2 + p2 ** 2 - 1.0)

    ps_dAdP11 = lambda p1, p2, p3: -(p1 * 2.0 - 2.0) / (p1 ** 2 + p2 ** 2 - 1.0) \
        + p1 * ((p1 - 1.0) ** 2 + p2 ** 2) / (p1 ** 2 + p2 ** 2 - 1.0) ** 2 * 2.0
    ps_dAdP12 = lambda p1, p2, p3: (p2 * -2.0) / (p1 ** 2 + p2 ** 2 - 1.0) \
        + p2 * ((p1 - 1.0) ** 2 + p2 ** 2) / (p1 ** 2 + p2 ** 2 - 1.0) ** 2 * 2.0
    ps_dAdP13 = lambda p1, p2, p3: torch.zeros_like(p1)

    ps_dAdP21 = lambda p1, p2, p3: p1 * p2 / (p1 ** 2 + p2 ** 2 - 1.0) ** 2 * -4.0
    ps_dAdP22 = lambda p1, p2, p3: 2.0 / (p1 ** 2 + p2 ** 2 - 1.0) \
        - p2 ** 2 / (p1 ** 2 + p2 ** 2 - 1.0) ** 2 * 4.0
    ps_dAdP23 = lambda p1, p2, p3: torch.zeros_like(p1)

    ps_dAdP31 = lambda p1, p2, p3: -(p1 * 2.0 + 2.0) / (p1 ** 2 + p2 ** 2 - 1.0) \
        + p1 * ((p1 + 1.0) ** 2 + p2 ** 2) / (p1 ** 2 + p2 ** 2 - 1.0) ** 2 * 2.0
    ps_dAdP32 = lambda p1, p2, p3: (p2 * -2.0) / (p1 ** 2 + p2 ** 2 - 1.0) \
        + p2 * ((p1 + 1.0) ** 2 + p2 ** 2) / (p1 ** 2 + p2 ** 2 - 1.0) ** 2 * 2.0
    ps_dAdP33 = lambda p1, p2, p3: torch.zeros_like(p1)

    fr1 = lambda p1, p2: p1 / torch.sqrt(p1 ** 2 + p2 ** 2) * ff(torch.sqrt(p1 ** 2 + p2 ** 2))
    fr2 = lambda p1, p2: p2 / torch.sqrt(p1 ** 2 + p2 ** 2) * ff(torch.sqrt(p1 ** 2 + p2 ** 2))

    # d( x/r f(r) )/dx = f (y^2/r^3) + g (x/r)(x/r), etc.
    pfr11 = lambda p1, p2: ff(torch.sqrt(p1 ** 2 + p2 ** 2)) * (p2 ** 2 / torch.sqrt(p1 ** 2 + p2 ** 2) ** 3) \
        + gg(torch.sqrt(p1 ** 2 + p2 ** 2)) * p1 ** 2 / (p1 ** 2 + p2 ** 2)
    pfr12 = lambda p1, p2: ff(torch.sqrt(p1 ** 2 + p2 ** 2)) * (-p1 * p2 / torch.sqrt(p1 ** 2 + p2 ** 2) ** 3) \
        + gg(torch.sqrt(p1 ** 2 + p2 ** 2)) * p1 * p2 / (p1 ** 2 + p2 ** 2)
    pfr21 = pfr12
    pfr22 = lambda p1, p2: ff(torch.sqrt(p1 ** 2 + p2 ** 2)) * (p1 ** 2 / torch.sqrt(p1 ** 2 + p2 ** 2) ** 3) \
        + gg(torch.sqrt(p1 ** 2 + p2 ** 2)) * p2 ** 2 / (p1 ** 2 + p2 ** 2)

    sym['mA1'] = lambda p1, p2, p3: pmA1(fr1(p1, p2), fr2(p1, p2), p3)
    sym['mA2'] = lambda p1, p2, p3: pmA2(fr1(p1, p2), fr2(p1, p2), p3)
    sym['mA3'] = lambda p1, p2, p3: pmA3(fr1(p1, p2), fr2(p1, p2), p3)

    np_ = 3
    sym['np'] = np_

    sym['s_dAdP1'] = [
        lambda p1, p2, p3: ps_dAdP11(fr1(p1, p2), fr2(p1, p2), p3) * pfr11(p1, p2)
        + ps_dAdP12(fr1(p1, p2), fr2(p1, p2), p3) * pfr21(p1, p2),
        lambda p1, p2, p3: ps_dAdP11(fr1(p1, p2), fr2(p1, p2), p3) * pfr12(p1, p2)
        + ps_dAdP12(fr1(p1, p2), fr2(p1, p2), p3) * pfr22(p1, p2),
        ps_dAdP13,
    ]
    sym['s_dAdP2'] = [
        lambda p1, p2, p3: ps_dAdP21(fr1(p1, p2), fr2(p1, p2), p3) * pfr11(p1, p2)
        + ps_dAdP22(fr1(p1, p2), fr2(p1, p2), p3) * pfr21(p1, p2),
        lambda p1, p2, p3: ps_dAdP21(fr1(p1, p2), fr2(p1, p2), p3) * pfr12(p1, p2)
        + ps_dAdP22(fr1(p1, p2), fr2(p1, p2), p3) * pfr22(p1, p2),
        ps_dAdP23,
    ]
    sym['s_dAdP3'] = [
        lambda p1, p2, p3: ps_dAdP31(fr1(p1, p2), fr2(p1, p2), p3) * pfr11(p1, p2)
        + ps_dAdP32(fr1(p1, p2), fr2(p1, p2), p3) * pfr21(p1, p2),
        lambda p1, p2, p3: ps_dAdP31(fr1(p1, p2), fr2(p1, p2), p3) * pfr12(p1, p2)
        + ps_dAdP32(fr1(p1, p2), fr2(p1, p2), p3) * pfr22(p1, p2),
        ps_dAdP33,
    ]

    def _nA1(p1, p2, p3):
        raise NotImplementedError('not implemented!')

    sym['nA1'] = _nA1
    sym['nA2'] = lambda p1, p2, p3: (p2 * -2.0) / (p1 ** 2 + p2 ** 2 - 1.0)
    sym['nA3'] = lambda p1, p2, p3: -((p1 - 1.0) ** 2 + p2 ** 2) / (p1 ** 2 + p2 ** 2 - 1.0)

    t_dAdP11 = lambda p1, p2, p3: -(p1 * 2.0 + 2.0) / (p1 ** 2 + p2 ** 2 - 1.0) \
        + p1 * ((p1 + 1.0) ** 2 + p2 ** 2) / (p1 ** 2 + p2 ** 2 - 1.0) ** 2 * 2.0
    t_dAdP12 = lambda p1, p2, p3: (p2 * -2.0) / (p1 ** 2 + p2 ** 2 - 1.0) \
        + p2 * ((p1 + 1.0) ** 2 + p2 ** 2) / (p1 ** 2 + p2 ** 2 - 1.0) ** 2 * 2.0
    t_dAdP13 = lambda p1, p2, p3: torch.zeros_like(p1)

    t_dAdP21 = lambda p1, p2, p3: p1 * p2 / (p1 ** 2 + p2 ** 2 - 1.0) ** 2 * 4.0
    t_dAdP22 = lambda p1, p2, p3: -2.0 / (p1 ** 2 + p2 ** 2 - 1.0) \
        + p2 ** 2 / (p1 ** 2 + p2 ** 2 - 1.0) ** 2 * 4.0
    t_dAdP23 = lambda p1, p2, p3: torch.zeros_like(p1)

    t_dAdP31 = lambda p1, p2, p3: -(p1 * 2.0 - 2.0) / (p1 ** 2 + p2 ** 2 - 1.0) \
        + p1 * ((p1 - 1.0) ** 2 + p2 ** 2) / (p1 ** 2 + p2 ** 2 - 1.0) ** 2 * 2.0
    t_dAdP32 = lambda p1, p2, p3: (p2 * -2.0) / (p1 ** 2 + p2 ** 2 - 1.0) \
        + p2 * ((p1 - 1.0) ** 2 + p2 ** 2) / (p1 ** 2 + p2 ** 2 - 1.0) ** 2 * 2.0
    t_dAdP33 = lambda p1, p2, p3: torch.zeros_like(p1)

    sym['t_dAdP1'] = [t_dAdP11, t_dAdP12, t_dAdP13]
    sym['t_dAdP2'] = [t_dAdP21, t_dAdP22, t_dAdP23]
    sym['t_dAdP3'] = [t_dAdP31, t_dAdP32, t_dAdP33]

    return sym
