"""res_sym_grad_complex_sym.m

A MATLAB *script* that defines a batch of elementwise anonymous functions in the
caller's workspace; here a function returning them in a dict, following
res_sym_grad_complex_plane_det1.py.

Unlike the other res_sym_grad modules these expressions are large, so the
closures are written as `def`s with named subexpressions rather than one-line
lambdas; the arithmetic is a literal transcription. `abs(p3).*sign(p3)` is kept
as-is (it equals p3 for real p3) so the formulas can be diffed against the .m.
"""

import torch


def res_sym_grad_complex_sym():
    sym = {}

    def _parts(p1, p2, p3):
        a3 = torch.abs(p3)
        s3 = torch.sign(p3)
        a3p = torch.abs(p3 + 1.0)
        s3p = torch.sign(p3 + 1.0)
        # the shared denominator of mA*
        E = -a3p ** 2 + p1 ** 2 + p2 ** 2
        return a3, s3, a3p, s3p, E

    def _mA1(p1, p2, p3):
        a3, _, _, _, E = _parts(p1, p2, p3)
        return -((p1 - 1.0) ** 2 - a3 ** 2 + p2 ** 2) / E

    def _mA2(p1, p2, p3):
        _, _, _, _, E = _parts(p1, p2, p3)
        return (p2 * 2.0) / E

    def _mA3(p1, p2, p3):
        a3, _, _, _, E = _parts(p1, p2, p3)
        return -((p1 + 1.0) ** 2 - a3 ** 2 + p2 ** 2) / E

    sym['mA1'] = _mA1
    sym['mA2'] = _mA2
    sym['mA3'] = _mA3

    sym['np'] = 3

    def _s_dAdP11(p1, p2, p3):
        a3, _, _, _, E = _parts(p1, p2, p3)
        N1 = (p1 - 1.0) ** 2 - a3 ** 2 + p2 ** 2
        return -(p1 * 2.0 - 2.0) / E + p1 * N1 / E ** 2 * 2.0

    def _s_dAdP12(p1, p2, p3):
        a3, _, _, _, E = _parts(p1, p2, p3)
        N1 = (p1 - 1.0) ** 2 - a3 ** 2 + p2 ** 2
        return (p2 * -2.0) / E + p2 * N1 / E ** 2 * 2.0

    def _s_dAdP13(p1, p2, p3):
        a3, s3, a3p, s3p, E = _parts(p1, p2, p3)
        N1 = (p1 - 1.0) ** 2 - a3 ** 2 + p2 ** 2
        return (a3 * s3 * 2.0) / E - a3p * s3p * N1 / E ** 2 * 2.0

    def _s_dAdP21(p1, p2, p3):
        _, _, _, _, E = _parts(p1, p2, p3)
        return p1 * p2 / E ** 2 * -4.0

    def _s_dAdP22(p1, p2, p3):
        _, _, _, _, E = _parts(p1, p2, p3)
        return 2.0 / E - p2 ** 2 / E ** 2 * 4.0

    def _s_dAdP23(p1, p2, p3):
        _, _, a3p, s3p, E = _parts(p1, p2, p3)
        return p2 * a3p * s3p / E ** 2 * 4.0

    def _s_dAdP31(p1, p2, p3):
        a3, _, _, _, E = _parts(p1, p2, p3)
        N3 = (p1 + 1.0) ** 2 - a3 ** 2 + p2 ** 2
        return -(p1 * 2.0 + 2.0) / E + p1 * N3 / E ** 2 * 2.0

    def _s_dAdP32(p1, p2, p3):
        a3, _, _, _, E = _parts(p1, p2, p3)
        N3 = (p1 + 1.0) ** 2 - a3 ** 2 + p2 ** 2
        return (p2 * -2.0) / E + p2 * N3 / E ** 2 * 2.0

    def _s_dAdP33(p1, p2, p3):
        a3, s3, a3p, s3p, E = _parts(p1, p2, p3)
        N3 = (p1 + 1.0) ** 2 - a3 ** 2 + p2 ** 2
        return (a3 * s3 * 2.0) / E - a3p * s3p * N3 / E ** 2 * 2.0

    sym['s_dAdP1'] = [_s_dAdP11, _s_dAdP12, _s_dAdP13]
    sym['s_dAdP2'] = [_s_dAdP21, _s_dAdP22, _s_dAdP23]
    sym['s_dAdP3'] = [_s_dAdP31, _s_dAdP32, _s_dAdP33]

    def _nparts(p1, p2, p3):
        a3, s3, a3p, s3p, E = _parts(p1, p2, p3)
        # Q is the shared denominator of nA*; Pp/Pm the two numerator factors.
        Q = a3 ** 2 * 2.0 - a3 ** 4 + p1 ** 2 * a3 ** 2 * 2.0 + p2 ** 2 * a3 ** 2 * 2.0 \
            + p1 ** 2 * 2.0 + p2 ** 2 * 2.0 - p1 ** 4 - p2 ** 4 \
            - p1 ** 2 * p2 ** 2 * 2.0 - 1.0
        Pp = p1 * 2.0 - a3 ** 2 + p1 ** 2 + p2 ** 2 + 1.0
        Pm = p1 * -2.0 - a3 ** 2 + p1 ** 2 + p2 ** 2 + 1.0
        dQ1 = p1 * 4.0 + p1 * a3 ** 2 * 4.0 - p1 * p2 ** 2 * 4.0 - p1 ** 3 * 4.0
        dQ2 = p2 * 4.0 + p2 * a3 ** 2 * 4.0 - p1 ** 2 * p2 * 4.0 - p2 ** 3 * 4.0
        dQ3 = a3 * s3 * 4.0 - a3 ** 3 * s3 * 4.0 + p1 ** 2 * a3 * s3 * 4.0 \
            + p2 ** 2 * a3 * s3 * 4.0
        return a3, s3, a3p, s3p, E, Q, Pp, Pm, dQ1, dQ2, dQ3

    def _nA1(p1, p2, p3):
        _, _, _, _, E, Q, Pp, _, _, _, _ = _nparts(p1, p2, p3)
        return (E * Pp) / Q

    def _nA2(p1, p2, p3):
        _, _, _, _, E, Q, _, _, _, _, _ = _nparts(p1, p2, p3)
        return (p2 * E * 2.0) / Q

    def _nA3(p1, p2, p3):
        _, _, _, _, E, Q, _, Pm, _, _, _ = _nparts(p1, p2, p3)
        return (E * Pm) / Q

    sym['nA1'] = _nA1
    sym['nA2'] = _nA2
    sym['nA3'] = _nA3

    def _t_dAdP11(p1, p2, p3):
        _, _, _, _, E, Q, Pp, _, dQ1, _, _ = _nparts(p1, p2, p3)
        return (p1 * Pp * 2.0) / Q + ((p1 * 2.0 + 2.0) * E) / Q \
            - E * dQ1 * Pp / Q ** 2

    def _t_dAdP12(p1, p2, p3):
        _, _, _, _, E, Q, Pp, _, _, dQ2, _ = _nparts(p1, p2, p3)
        return (p2 * E * 2.0) / Q + (p2 * Pp * 2.0) / Q - E * dQ2 * Pp / Q ** 2

    def _t_dAdP13(p1, p2, p3):
        a3, s3, a3p, s3p, E, Q, Pp, _, _, _, dQ3 = _nparts(p1, p2, p3)
        return (a3 * s3 * E * -2.0) / Q - (a3p * s3p * Pp * 2.0) / Q \
            - E * dQ3 * Pp / Q ** 2

    def _t_dAdP21(p1, p2, p3):
        _, _, _, _, E, Q, _, _, dQ1, _, _ = _nparts(p1, p2, p3)
        return (p1 * p2 * 4.0) / Q - p2 * E * dQ1 / Q ** 2 * 2.0

    def _t_dAdP22(p1, p2, p3):
        _, _, _, _, E, Q, _, _, _, dQ2, _ = _nparts(p1, p2, p3)
        return (p2 ** 2 * 4.0) / Q + (E * 2.0) / Q - p2 * E * dQ2 / Q ** 2 * 2.0

    def _t_dAdP23(p1, p2, p3):
        _, _, a3p, s3p, E, Q, _, _, _, _, dQ3 = _nparts(p1, p2, p3)
        return (p2 * a3p * s3p * -4.0) / Q - p2 * E * dQ3 / Q ** 2 * 2.0

    def _t_dAdP31(p1, p2, p3):
        _, _, _, _, E, Q, _, Pm, dQ1, _, _ = _nparts(p1, p2, p3)
        return (p1 * Pm * 2.0) / Q + ((p1 * 2.0 - 2.0) * E) / Q \
            - E * dQ1 * Pm / Q ** 2

    def _t_dAdP32(p1, p2, p3):
        _, _, _, _, E, Q, _, Pm, _, dQ2, _ = _nparts(p1, p2, p3)
        return (p2 * E * 2.0) / Q + (p2 * Pm * 2.0) / Q - E * dQ2 * Pm / Q ** 2

    def _t_dAdP33(p1, p2, p3):
        a3, s3, a3p, s3p, E, Q, _, Pm, _, _, dQ3 = _nparts(p1, p2, p3)
        return (a3 * s3 * E * -2.0) / Q - (a3p * s3p * Pm * 2.0) / Q \
            - E * dQ3 * Pm / Q ** 2

    sym['t_dAdP1'] = [_t_dAdP11, _t_dAdP12, _t_dAdP13]
    sym['t_dAdP2'] = [_t_dAdP21, _t_dAdP22, _t_dAdP23]
    sym['t_dAdP3'] = [_t_dAdP31, _t_dAdP32, _t_dAdP33]

    return sym
