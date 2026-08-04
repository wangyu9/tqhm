"""oracle_conjugate_newton_symmetric.m

Value and gradient of the quasi-harmonic energy with respect to the tensor
variables `at`. The map (u,v) is not a free variable: it is the solution of the
anisotropic Laplace equation for the current tensor field, so the gradient
already accounts for that inner solve (the `conjugate` part of the name) and no
autodiff is needed.

Follows the `old_imp == false` branch of the MATLAB source; the `old_imp == true`
branch is preserved in oracle_conjugate_newton_symmetric_old.py.
"""

import numpy as np
import torch

from tqhm_config import DEV, DT, td, ti, get_verbose
from tensor_para_faster import tensor_para_faster
from assemble_lap_core import assemble_lap_core
from lap_submatrix import LapBlocks
from tdss_solver import ReusableSPDSolver
from sparse_torch import SpOp


def _span_dot_xy(bbx, bby, ccx, ccy):
    """symmetric_tensor_span_dot for d==2, laid out as [11; 12+21; 22]."""
    return torch.cat([bbx * ccx, bby * ccx + bbx * ccy, bby * ccy])


def oracle_conjugate_newton_symmetric(mesh, at, BCBN, tp, reuse):
    V = mesh['V']
    F = mesh['F']
    n = V.shape[0]

    BC = BCBN[:, 0:2]
    BN = BCBN[:, 2:4]

    alpha = tp['alpha']
    beta = tp['beta']
    gamma = tp['gamma']

    f = F.shape[0]
    if at.shape[0] != f:
        at = at.reshape(3, -1).t().contiguous() if at.dim() == 1 else at
    if at.dim() == 1:
        at = at.reshape(3, f).t().contiguous()

    out = {'stop_sign': False}

    Area = mesh['AI']
    known = mesh['IKB']
    unknown = mesh['IUB']

    assert tp['conj_vmap'] is False

    uvc = torch.complex(V[:, 0].clone(), V[:, 1].clone())
    uvc[known] = torch.complex(BC[:, 0], BC[:, 1])

    # --- tensor parameterization ---
    au, paupat = tensor_para_faster(at, tp['para_type'])
    au = au * Area[:, None]

    # --- one-time structure setup (mirrors the isempty(reuse.RL) branch) ---
    if reuse.RL is None:
        Fnp = mesh['F_np']
        reuse.RL = assemble_lap_core(n, Fnp)

        reuse.sub = LapBlocks(n, reuse.RL.full.indptr, reuse.RL.full.indices,
                              unknown, known)
        reuse.solver = ReusableSPDSolver(
            reuse.sub.indptr_uu, reuse.sub.indices_uu, reuse.sub.nu, batch_size=2
        )

        GI_sp = mesh['GI_sp']
        Gx = SpOp(GI_sp[:f, :])
        Gy = SpOp(GI_sp[f:2 * f, :])
        reuse.grad_xy = lambda uu: (Gx.matvec(uu), Gy.matvec(uu))
        reuse.div_xy = lambda ux, uy: Gx.rmatvec(ux) + Gy.rmatvec(uy)

    grad_xy = reuse.grad_xy
    div_xy = reuse.div_xy

    a11, a12, a22 = au[:, 0], au[:, 1], au[:, 2]

    # values of the full Laplacian on the fixed sparsity pattern
    data = reuse.RL.asb_full(mesh['GIS'], a11, a12, a22)

    Auu_data = reuse.sub.Auu_data(data)
    rhs = -reuse.sub.Auk_matvec(data, uvc[known])

    uvc_u = reuse.solver.solve_complex(Auu_data, rhs)
    uvc = uvc.clone()
    uvc[unknown] = uvc_u

    Gx_uvc, Gy_uvc = grad_xy(uvc)

    Gxu = Gx_uvc.real
    Gyu = Gy_uvc.real
    Gxv = Gx_uvc.imag
    Gyv = Gy_uvc.imag

    gn_u = 0.5 * _span_dot_xy(Gxu, Gyu, Gxu, Gyu)
    gn_v = 0.5 * _span_dot_xy(Gxv, Gyv, Gxv, Gyv)

    GtAGuvc = div_xy(
        a11.to(torch.complex128) * Gx_uvc + a12.to(torch.complex128) * Gy_uvc,
        a12.to(torch.complex128) * Gx_uvc + a22.to(torch.complex128) * Gy_uvc,
    )

    out['newArea'] = Area * (Gxu * Gyv - Gxv * Gyu)
    flipped = out['newArea'] < 0
    out['num_flipped'] = int(torch.count_nonzero(flipped).item())
    out['stop_sign'] = out['num_flipped'] == 0

    g_drlt = paupat['s_pdapdt'](gn_u + gn_v, Area)

    # E = 0.5 * uvc^H (G' A G) uvc - 0.5 * <BC, BN>
    E_drlt = 0.5 * torch.vdot(uvc, GtAGuvc).real \
        - 0.5 * torch.dot(BC[:, 0], BN[:, 0]) \
        - 0.5 * torch.dot(BC[:, 1], BN[:, 1])

    g_bn = 0.0
    E_bn = 0.0
    if beta != 0:
        raise NotImplementedError('beta term not implemented in this branch')

    g_wbn = 0.0
    E_wbn = 0.0
    if gamma != 0:
        raise NotImplementedError('gamma term not implemented in this branch')

    u = uvc.real.contiguous()
    v = uvc.imag.contiguous()

    E = alpha * E_drlt + E_bn * beta + E_wbn * gamma
    g = alpha * g_drlt + g_bn * beta + g_wbn * gamma

    if get_verbose() >= 2:
        print("\t\t E_drlt=%g,\t E_bn=%g,\t E_wbn=%g,\t " % (
            _sc(E_drlt), _sc(E_bn), _sc(E_wbn)), end='')

    if tp.get('energy_uv') is not None:
        er_uv, r_uv = tp['energy_uv'](torch.stack([u, v], dim=1))
        r_u = r_uv[:n]
        r_v = r_uv[n:2 * n]

        rc = torch.complex(r_u, r_v)
        sol = torch.zeros(n, dtype=torch.complex128, device=DEV)
        sol[unknown] = reuse.solver.solve_complex(Auu_data, rc[unknown])
        trx_uvc, try_uvc = grad_xy(sol)

        gr_uv = paupat['s_pdapdt'](
            -_span_dot_xy(Gxu, Gyu, trx_uvc.real, try_uvc.real)
            - _span_dot_xy(Gxv, Gyv, trx_uvc.imag, try_uvc.imag),
            Area,
        )
        E = E + er_uv
        g = g + gr_uv
        if get_verbose() >= 2:
            print("Er_uv=%g,\t \t" % _sc(er_uv), end='')
    elif get_verbose() >= 2:
        print()

    reg = tp['reg_value'](at)
    if _sc(tp['rau_value'](au)) != 0:
        raise NotImplementedError('not implemented!')

    if get_verbose() >= 2:
        print("Reg=%g,\t \t" % _sc(reg), end='')

    E = E + reg
    g = g + tp['reg_grad'](at)

    Hess = None
    out['u'] = u
    out['v'] = v

    if reuse.fine_recorder is not None:
        if reuse.fine_recorder is True:
            reuse.fine_recorder = []
        reuse.fine_recorder.append({'u': u, 'v': v})

    return E, g, Hess, out


def _sc(x):
    if torch.is_tensor(x):
        return float(x.item()) if x.numel() == 1 else float(x.reshape(-1)[0].item())
    return float(x)
