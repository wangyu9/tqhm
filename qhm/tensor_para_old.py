"""tensor_para_old.m -- the earlier tensor_para, superseded by tensor_para.py.

Note the .m file declares `function [tp] = tensor_para(...)`, i.e. it shadows
tensor_para under a different filename; the Python function is named after the
file. It predates the `np`-generic cell arrays: the nine s_dAdP / t_dAdP entries
are addressed by flat name, only 3 parameters are supported, and there is no
regularizer, no mass-matrix-free branch, no Hessian and no symbolic option.

The big `if false` symbolic-derivation block is preserved as `if False:`, with a
NotImplementedError since the Symbolic Math Toolbox is not available here (same
choice as tensor_para.py's `experimental` branch).
"""

import numpy as np
import scipy.sparse as sp
import torch

from tqhm_config import DT
from default_args import Args
from res_sym_grad_both_inverse import res_sym_grad_both_inverse
from res_sym_grad_complex_sym import res_sym_grad_complex_sym
from res_sym_grad_complex_sym_det1 import res_sym_grad_complex_sym_det1
from res_sym_grad_diag import res_sym_grad_diag
from res_sym_grad_diag_sq import res_sym_grad_diag_sq
from res_sym_grad_direct import res_sym_grad_direct
from res_sym_grad_llt import res_sym_grad_llt
from res_sym_grad_weber import res_sym_grad_weber


def tensor_para_old(FA, dim, para_type):
    tp = Args()

    tp.reg_grad = lambda at: 0
    tp.append_grad = torch.zeros(0, dtype=DT)

    tp.conj_vmap = True

    Area = FA
    f = FA.shape[0]

    assert dim == 2

    if False:
        if False:
            # diag without mass matrix.

            s_at2au = lambda aatt: torch.cat(
                [aatt[:, 0], torch.zeros(f, dtype=DT, device=aatt.device), aatt[:, 2]])
            s_pdapdt_lmul = lambda aatt, gg: torch.cat(
                [torch.ones(f, dtype=DT), torch.zeros(f, dtype=DT), torch.ones(f, dtype=DT)]) * gg

            s_at2au_v = lambda aatt: torch.cat(
                [1 / aatt[:, 0], torch.zeros(f, dtype=DT), 1 / aatt[:, 2]])
            s_pdapdt_lmul_v = lambda aatt, gg: torch.cat(
                [-1 / aatt[:, 0] ** 2, torch.zeros(f, dtype=DT), -1 / aatt[:, 2] ** 2]) * gg
        else:
            # with mass matrix.
            # this is for diagonal tensor.

            s_at2au = lambda aatt: torch.cat(
                [Area * aatt[:, 0], torch.zeros(f, dtype=DT), Area * aatt[:, 2]])
            s_pdapdt_lmul = lambda aatt, gg: torch.cat(
                [Area, torch.zeros(f, dtype=DT), Area]) * gg

            s_at2au_v = lambda aatt: torch.cat(
                [Area / aatt[:, 0], torch.zeros(f, dtype=DT), Area / aatt[:, 2]])
            s_pdapdt_lmul_v = lambda aatt, gg: torch.cat(
                [-Area / aatt[:, 0] ** 2, torch.zeros(f, dtype=DT), -Area / aatt[:, 2] ** 2]) * gg

    if False:
        # The `syms`/`matlabFunction` derivation of every para_type, whose printed
        # output is exactly what the res_sym_grad_* modules contain.
        raise NotImplementedError(
            'the Symbolic Math Toolbox branch of tensor_para_old.m is not ported; '
            'the derived closures live in the res_sym_grad_* modules')
    else:
        if para_type == 'diag-no-mass':
            sym = {}
        elif para_type == 'diag':
            sym = res_sym_grad_diag()
        elif para_type == 'diag-sq':
            sym = res_sym_grad_diag_sq()
        elif para_type == 'complex-sym':
            sym = res_sym_grad_complex_sym()
        elif para_type == 'complex-sym-det1':
            sym = res_sym_grad_complex_sym_det1()
        elif para_type == 'both-inverse':
            sym = res_sym_grad_both_inverse()
        elif para_type == 'llt':
            sym = res_sym_grad_llt()
        elif para_type == 'weber':
            sym = res_sym_grad_weber()
        elif para_type == 'direct':
            sym = res_sym_grad_direct()
        else:
            raise NotImplementedError('Unsupported para type~ (%s)' % para_type)

        mA1, mA2, mA3 = sym['mA1'], sym['mA2'], sym['mA3']
        nA1, nA2, nA3 = sym['nA1'], sym['nA2'], sym['nA3']

        s_dAdP11, s_dAdP12, s_dAdP13 = sym['s_dAdP1']
        s_dAdP21, s_dAdP22, s_dAdP23 = sym['s_dAdP2']
        s_dAdP31, s_dAdP32, s_dAdP33 = sym['s_dAdP3']

        t_dAdP11, t_dAdP12, t_dAdP13 = sym['t_dAdP1']
        t_dAdP21, t_dAdP22, t_dAdP23 = sym['t_dAdP2']
        t_dAdP31, t_dAdP32, t_dAdP33 = sym['t_dAdP3']

        _ap = lambda fun: (lambda aatt: fun(aatt[:, 0], aatt[:, 1], aatt[:, 2]))

        s_11 = _ap(s_dAdP11)
        s_21 = _ap(s_dAdP21)
        s_31 = _ap(s_dAdP31)

        s_12 = _ap(s_dAdP12)
        s_22 = _ap(s_dAdP22)
        s_32 = _ap(s_dAdP32)

        s_13 = _ap(s_dAdP13)
        s_23 = _ap(s_dAdP23)
        s_33 = _ap(s_dAdP33)

        t_11 = _ap(t_dAdP11)
        t_21 = _ap(t_dAdP21)
        t_31 = _ap(t_dAdP31)

        t_12 = _ap(t_dAdP12)
        t_22 = _ap(t_dAdP22)
        t_32 = _ap(t_dAdP32)

        t_13 = _ap(t_dAdP13)
        t_23 = _ap(t_dAdP23)
        t_33 = _ap(t_dAdP33)

        if False:
            idf = np.arange(f)
            idfau = np.arange(3 * f)

            def s_pdapdt(aatt):
                rows = np.concatenate([np.tile(idf + 0 * f, 3),
                                       np.tile(idf + 1 * f, 3),
                                       np.tile(idf + 2 * f, 3)])
                cols = np.tile(idfau, 3)
                A = np.asarray(Area.detach().cpu())
                vals = np.concatenate([
                    np.concatenate([np.asarray(s_11(aatt).detach().cpu()) * A,
                                    np.asarray(s_12(aatt).detach().cpu()) * A,
                                    np.asarray(s_13(aatt).detach().cpu()) * A]),
                    np.concatenate([np.asarray(s_21(aatt).detach().cpu()) * A,
                                    np.asarray(s_22(aatt).detach().cpu()) * A,
                                    np.asarray(s_23(aatt).detach().cpu()) * A]),
                    np.concatenate([np.asarray(s_31(aatt).detach().cpu()) * A,
                                    np.asarray(s_32(aatt).detach().cpu()) * A,
                                    np.asarray(s_33(aatt).detach().cpu()) * A]),
                ])
                return sp.coo_matrix((vals, (rows, cols)),
                                     shape=(3 * f, idfau.size)).tocsr()

            # not "*" here, critical to have the transpose here.
            s_pdapdt_lmul = lambda aatt, gg: s_pdapdt(aatt).T @ gg
            # this somehows gives incorrect result, check it before use!!!

        s_at2au = lambda aatt: torch.cat([
            Area * mA1(aatt[:, 0], aatt[:, 1], aatt[:, 2]),
            Area * mA2(aatt[:, 0], aatt[:, 1], aatt[:, 2]),
            Area * mA3(aatt[:, 0], aatt[:, 1], aatt[:, 2]),
        ])

        s_pdapdt_lmul = lambda aatt, gg: torch.cat([
            Area * (s_11(aatt) * gg[0:f] + s_21(aatt) * gg[f:2 * f] + s_31(aatt) * gg[2 * f:3 * f]),
            Area * (s_12(aatt) * gg[0:f] + s_22(aatt) * gg[f:2 * f] + s_32(aatt) * gg[2 * f:3 * f]),
            Area * (s_13(aatt) * gg[0:f] + s_23(aatt) * gg[f:2 * f] + s_33(aatt) * gg[2 * f:3 * f]),
        ])

        # this is for diagonal tensor:
        # s_pdapdt_lmul = lambda aatt, gg: torch.cat([Area, zeros(f), Area]) * gg

        # s_at2au_v = lambda aatt: torch.cat([Area/aatt[:,0], zeros(f), Area/aatt[:,2]])
        # s_pdapdt_lmul_v = lambda aatt, gg: torch.cat(
        #     [-Area/aatt[:,0]**2, zeros(f), -Area/aatt[:,2]**2]) * gg

        s_at2au_v = lambda aatt: torch.cat([
            Area * nA1(aatt[:, 0], aatt[:, 1], aatt[:, 2]),
            Area * nA2(aatt[:, 0], aatt[:, 1], aatt[:, 2]),
            Area * nA3(aatt[:, 0], aatt[:, 1], aatt[:, 2]),
        ])

        s_pdapdt_lmul_v = lambda aatt, gg: torch.cat([
            Area * (t_11(aatt) * gg[0:f] + t_21(aatt) * gg[f:2 * f] + t_31(aatt) * gg[2 * f:3 * f]),
            Area * (t_12(aatt) * gg[0:f] + t_22(aatt) * gg[f:2 * f] + t_32(aatt) * gg[2 * f:3 * f]),
            Area * (t_13(aatt) * gg[0:f] + t_23(aatt) * gg[f:2 * f] + t_33(aatt) * gg[2 * f:3 * f]),
        ])

    tp.s_at2au = s_at2au
    tp.s_pdapdt_lmul = s_pdapdt_lmul

    tp.s_at2au_v = s_at2au_v
    tp.s_pdapdt_lmul_v = s_pdapdt_lmul_v

    return tp
