"""tensor_para.m -- build the tensor-parameterization struct `tp`.

Only the `experimental == false` path is ported: the Symbolic-Math-Toolbox branch
(`if experimental`) derives the same closures at run time with `syms`/`diff` and
is replaced here by the pre-derived res_sym_grad_* scripts, exactly as the
MATLAB `else` branch does.

Note on `t_dAdP1{ii}`: the res_sym_grad_* scripts define the flat names
t_dAdP11..t_dAdP33 rather than cell arrays, so MATLAB's `t_dAdP1{ii}` only works
for the symbolic branch. Here they are grouped into lists, which is the evident
intent, and they are unused by the solver anyway (`s_at2au_v` /
`s_pdapdt_lmul_v` are only for the 'both-inverse' style parameterizations).
"""

import numpy as np
import torch

from tqhm_config import DT
from default_args import Args
from res_sym_grad_both_inverse import res_sym_grad_both_inverse
from res_sym_grad_complex_plane_det1 import res_sym_grad_complex_plane_det1
from res_sym_grad_complex_sym import res_sym_grad_complex_sym
from res_sym_grad_complex_sym_det1 import res_sym_grad_complex_sym_det1
from res_sym_grad_diag import res_sym_grad_diag
from res_sym_grad_diag_sq import res_sym_grad_diag_sq
from res_sym_grad_direct import res_sym_grad_direct
from res_sym_grad_llt import res_sym_grad_llt
from res_sym_grad_weber import res_sym_grad_weber


def _sparse_diag(ddd):
    import scipy.sparse as sp
    d = np.asarray(ddd, dtype=np.float64).ravel()
    return sp.diags(d, 0, shape=(d.size, d.size), format='csr')


def _apf(fun, aatt):
    """tensor_para.m/apf: apply an elementwise closure to the columns of `at`."""
    m = aatt.shape[1]
    assert m >= 3
    if m == 3:
        return fun(aatt[:, 0], aatt[:, 1], aatt[:, 2])
    return fun(*[aatt[:, k] for k in range(m)])


def _assemble_pdapdt(s_1, s_2, s_3, aatt, gg, Area):
    """tensor_para.m/assemble_pdapdt."""
    f = Area.shape[0]
    m = aatt.shape[1]
    assert aatt.shape[0] == f
    gg = gg.reshape(3, -1).t() if gg.dim() == 1 else gg
    out = []
    for ii in range(m):
        out.append(Area * (s_1[ii](aatt) * gg[:, 0]
                           + s_2[ii](aatt) * gg[:, 1]
                           + s_3[ii](aatt) * gg[:, 2]))
    return torch.cat(out)


def _assemble_pdrpdt(r, aatt, Area):
    """tensor_para.m/assemble_pdrpdt."""
    f = Area.shape[0]
    m = aatt.shape[1]
    assert aatt.shape[0] == f
    return torch.cat([Area * r[ii](aatt) for ii in range(m)])


def tensor_para(FA, dim, para_type, RegCoeff=None, InversePara=False,
                Experimental=False):
    reg_coeff = 0 if RegCoeff is None else RegCoeff   # not yet used anyway
    inv_para_fun = InversePara
    experimental = Experimental                        # much slower branch

    tp = Args()
    tp.para_type = para_type
    tp.test_mark = False

    tp.alpha = 1
    tp.beta = 0
    tp.gamma = 0

    # isempty(tp.energy_uv) decides whether to add the tp.energy_uv term
    tp.energy_uv = None

    # regularizer in at
    tp.reg_value = lambda at: 0
    tp.reg_grad = lambda at: 0

    # regularizer in au
    tp.rau_value = lambda au: 0
    tp.rau_grad = lambda au: 0

    tp.pre_cond = None
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
            # with mass matrix; this is for a diagonal tensor.
            s_at2au = lambda aatt: torch.cat(
                [Area * aatt[:, 0], torch.zeros(f, dtype=DT), Area * aatt[:, 2]])
            s_pdapdt_lmul = lambda aatt, gg: torch.cat(
                [Area, torch.zeros(f, dtype=DT), Area]) * gg

            s_at2au_v = lambda aatt: torch.cat(
                [Area / aatt[:, 0], torch.zeros(f, dtype=DT), Area / aatt[:, 2]])
            s_pdapdt_lmul_v = lambda aatt, gg: torch.cat(
                [-Area / aatt[:, 0] ** 2, torch.zeros(f, dtype=DT), -Area / aatt[:, 2] ** 2]) * gg

    if experimental:
        raise NotImplementedError(
            'the Symbolic Math Toolbox branch of tensor_para.m is not ported; '
            'the derived closures live in the res_sym_grad_* modules')

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
    elif para_type == 'complex-plane-det1':
        ff = lambda rr: 1 - 1 / (1 + torch.log(1 + rr))
        gg_ = lambda rr: 1 / (1 + torch.log(1 + rr)) ** 2 / (1 + rr)
        sym = res_sym_grad_complex_plane_det1(ff, gg_)
    elif para_type == 'complex-plane-det1-tanh':
        ff = lambda rr: torch.tanh(rr)
        gg_ = lambda rr: 1 - torch.tanh(rr) ** 2
        sym = res_sym_grad_complex_plane_det1(ff, gg_)
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

    np_ = sym['np']

    s_dAdP1 = sym['s_dAdP1']
    s_dAdP2 = sym['s_dAdP2']
    s_dAdP3 = sym['s_dAdP3']
    t_dAdP1 = sym['t_dAdP1']
    t_dAdP2 = sym['t_dAdP2']
    t_dAdP3 = sym['t_dAdP3']

    # s_dRdP is not used anyway (only defined by the symbolic branch).
    s_dRdP = sym.get('s_dRdP')
    mR = sym.get('mR')

    r = [(lambda aatt, ii=ii: _apf(s_dRdP[ii], aatt)) for ii in range(np_)] \
        if s_dRdP is not None else None

    s_1 = [(lambda aatt, ii=ii: _apf(s_dAdP1[ii], aatt)) for ii in range(np_)]
    s_2 = [(lambda aatt, ii=ii: _apf(s_dAdP2[ii], aatt)) for ii in range(np_)]
    s_3 = [(lambda aatt, ii=ii: _apf(s_dAdP3[ii], aatt)) for ii in range(np_)]

    t_1 = [(lambda aatt, ii=ii: _apf(t_dAdP1[ii], aatt)) for ii in range(np_)]
    t_2 = [(lambda aatt, ii=ii: _apf(t_dAdP2[ii], aatt)) for ii in range(np_)]
    t_3 = [(lambda aatt, ii=ii: _apf(t_dAdP3[ii], aatt)) for ii in range(np_)]

    if mR is not None:
        tp.s_at2reg = lambda aatt: torch.sum(Area * _apf(mR, aatt))
        tp.s_pdrpdt = lambda aatt: _assemble_pdrpdt(r, aatt, Area)
    else:
        tp.s_at2reg = None
        tp.s_pdrpdt = None

    s_at2au = lambda aatt: torch.cat([
        Area * _apf(sym['mA1'], aatt),
        Area * _apf(sym['mA2'], aatt),
        Area * _apf(sym['mA3'], aatt),
    ])

    s_pdapdt_lmul = lambda aatt, gg: _assemble_pdapdt(s_1, s_2, s_3, aatt, gg, Area)

    # this one asserts two parameters per triangle.
    def _s_pdapdt(aatt):
        import scipy.sparse as sp
        A = np.asarray(Area.detach().cpu())
        blk = lambda s, k: _sparse_diag(np.asarray(s[k](aatt).detach().cpu()) * A)
        return sp.bmat([[blk(s_1, 0), blk(s_1, 1)],
                        [blk(s_2, 0), blk(s_2, 1)],
                        [blk(s_3, 0), blk(s_3, 1)]], format='csr')

    tp.s_pdapdt = _s_pdapdt

    s_at2au_v = lambda aatt: torch.cat([
        Area * _apf(sym['nA1'], aatt),
        Area * _apf(sym['nA2'], aatt),
        Area * _apf(sym['nA3'], aatt),
    ])

    s_pdapdt_lmul_v = lambda aatt, gg: _assemble_pdapdt(t_1, t_2, t_3, aatt, gg, Area)

    tp.s_at2au = s_at2au
    tp.s_pdapdt_lmul = s_pdapdt_lmul

    tp.s_at2au_v = s_at2au_v
    tp.s_pdapdt_lmul_v = s_pdapdt_lmul_v

    return tp
