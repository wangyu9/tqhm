"""core_variational_beltrami.m -- the main solver script.

MATLAB script operating on the caller's workspace (`folder`, `args`); here a
function returning the same `out` struct. The `if false ... end` blocks (render,
plain adam, the local-global sandbox, the double-checks, save) are preserved
as `if False:` so they remain available.
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

from tqhm_config import DEV, DT, td, ti, npy
from readOBJ import readOBJ
from upsample import upsample
from doublearea import doublearea
from normalize_mesh import normalize_mesh
from triangle_mesh_basic import triangle_mesh_basic
from intrinsic_grad_equilateral import intrinsic_grad_equilateral
from tensor_para import tensor_para
from tensor_para_faster import tensor_para_faster
from reuse_struct import reuse_struct
from oracle_conjugate_newton_symmetric import oracle_conjugate_newton_symmetric
from core_optimize_block import core_optimize_block
from attempt_local_global import attempt_local_global
from sparse_torch import SpOp


def _sparse_diag(ddd):
    import scipy.sparse as sp
    d = np.asarray(ddd, dtype=np.float64).ravel()
    return sp.diags(d, 0, shape=(d.size, d.size), format='csr')


def core_variational_beltrami(folder, args, V=None, F=None, VT=None, da0=None):
    if args.load_format == 'du2020':
        # for [Du et al. 2020]
        V, F, VT, _, _, _ = readOBJ(str(folder) + '/input.obj')
        VT = VT[:, 0:2]

    elif args.load_format == 'du2020-old':
        # for [Du et al. 2020]
        V, F, _, _, _, _ = readOBJ(str(folder) + '/input.obj')
        VT, _, _, _, _, _ = readOBJ(str(folder) + '/result.obj')
        VT = VT[:, 0:2]

    elif args.load_format == 'none':
        pass

    elif args.load_format == 'du2020-proj':
        V, F, _, _, _, _ = readOBJ(str(folder) + '/input.obj')
        VT = V[:, 0:2]

    else:
        raise ValueError('unknown format!')

    # input: V, F, VT. For VT only the values on the boundary matter.
    # VT: target boundary vertices.
    assert V.shape[0] == VT.shape[0]
    assert VT.shape[1] == 2

    # normalize target area.
    if args.load_normalize_target:
        VT = normalize_mesh(V, F, VT)

    tt = args.rotate
    VT = VT @ np.array([[np.cos(tt), -np.sin(tt)], [np.sin(tt), np.cos(tt)]]).T

    for _ in range(args.sub_div_level):
        F_old = F
        V, F = upsample(V, F_old)
        VT, F2 = upsample(VT, F_old)

    out = {'finish_with_local_global': False}

    mesh = triangle_mesh_basic(V, F, args.indEC)

    known = mesh['IKB_np']
    unknown = mesh['IUB_np']

    TVB = VT[known, :]

    n = V.shape[0]
    f = F.shape[0]

    # NVT = D @ (R @ [TVB1, -TVB0]); R scatters boundary rows into an n-vector,
    # R.T @ NVT gathers them back.
    rhs = torch.zeros(n, 2, dtype=DT, device=DEV)
    rhs[ti(known)] = torch.stack([td(TVB[:, 1]), td(-TVB[:, 0])], dim=1)
    NVT = torch.sparse.mm(mesh['D'], rhs)      # (n, 2)
    # the old code used the target Laplacian; do not use any more:
    # NVT = -cotmatrix(VT, mesh.F) @ VT   (cotmatrix is negative definite)

    BC = TVB
    BN = npy(NVT[ti(known)])

    # the total area is 0.5*(BC[:,0]@BN[:,0] + BC[:,1]@BN[:,1]), which upon
    # convergence agrees with 0.5*sum(doublearea([u,v],F))

    Area = mesh['FA']
    mesh['FA'] = None   # to keep any code from using mesh.FA directly

    if args.epsilon_gradarea_angle is not None:
        from intrinsic_grad_area import intrinsic_grad_area
        GI, Area, GIS = intrinsic_grad_area(V, F, args.epsilon_gradarea_angle)
        mesh['GI_sp'] = GI
        mesh['GI'] = SpOp(GI)
        mesh['GIS_np'] = GIS
        mesh['GIS'] = {k: td(v) for k, v in GIS.items()}

    if args.graph_tutte:
        print('Init with uniform graph Laplacian')
        GI, GIS = intrinsic_grad_equilateral(n, F)
        mesh['GI_sp'] = GI
        mesh['GI'] = SpOp(GI)
        mesh['GIS_np'] = GIS
        mesh['GIS'] = {k: td(v) for k, v in GIS.items()}
        Area = np.ones_like(Area)

    mesh['AI'] = td(Area)
    Area_t = mesh['AI']

    # --- an optional nonlinear energy on (u,v) ---
    int_support_type = ['area', 'mass-spring', 'symmetric-Dirichlet', 'ARAP',
                        'symmetric-Dirichlet-capped', 'area-change']

    e_value_grad = None
    if args.energy_type != 'none':
        if any(args.energy_type in t or t in args.energy_type
               for t in int_support_type):
            # qhm/intrinsic-extrinsic/ is not a valid identifier, so it cannot be
            # imported as a package; put it on sys.path like the MATLAB path.
            _ie = str(Path(__file__).resolve().parent / 'intrinsic-extrinsic')
            if _ie not in sys.path:
                sys.path.insert(0, _ie)

            from intrinsic_grad_hessian import intrinsic_grad_hessian
            from IntrinsicHessianClass import IntrinsicHessianClass

            energy_ef, energy_gf, energy_hf = intrinsic_grad_hessian(args.energy_type)

            if args.energy_Wf is None:
                e_value_grad = lambda FW: (
                    IntrinsicHessianClass.Value(FW[:, 0:2], F, V, energy_ef),
                    IntrinsicHessianClass.Grad(FW[:, 0:2], F, V, energy_gf),
                )
            else:
                e_value_grad = lambda FW: (
                    IntrinsicHessianClass.ValueWithWeights(FW[:, 0:2], F, V, energy_ef, args.energy_Wf),
                    IntrinsicHessianClass.GradWithWeights(FW[:, 0:2], F, V, energy_gf, args.energy_Wf),
                )
        elif True:
            from OptimProblemArap import OptimProblemArap
            assert V.shape[1] == 2 or np.linalg.norm(V[:, 2]) == 0
            energy_obj = OptimProblemArap(V[:, 0:2], F, None, None, V[:, 0:2])
            e_value_grad = lambda FW: energy_obj.evaluateFunctional(
                FW.reshape(-1), True, True, False)

    # --- complex-plane-det1: equivalent ---
    da = torch.zeros(3 * f, dtype=DT, device=DEV)
    if da0 is None:
        # MATLAB: rng(1,'philox'); da = (rand([f,3])-0.5)*0.001;
        # Philox is not reproduced here, so the seed differs (see README note).
        g = torch.Generator(device='cpu').manual_seed(1)
        da = (torch.rand(f, 3, generator=g, dtype=DT) - 0.5) * 0.001
        da = da.to(DEV)
    else:
        da = td(da0).reshape(f, 3)

    LB = None
    UB = None
    NONLCON = None

    if args.L2_reg is None:
        tp = tensor_para(Area_t, 2, args.tp_type)   # 'complex-plane-det1'
        tp.reg_value = lambda at: args.reg_value(at, Area_t)
        tp.reg_grad = lambda at: args.reg_grad(at, Area_t)
    else:
        tp = tensor_para(Area_t, 2, args.tp_type, RegCoeff=0)  # not using it here
        tp.reg_grad = lambda at: args.L2_reg * torch.cat(
            [Area_t * at[:, 0], Area_t * at[:, 1], 0 * at[:, 2]])
        tp.reg_value = lambda at: args.L2_reg * 0.5 * torch.sum(
            Area_t * (at[:, 0] ** 2 + at[:, 1] ** 2))

    tp.conj_vmap = False   # True

    tp.alpha = args.alpha
    tp.beta = args.beta
    tp.gamma = args.gamma

    if args.energy_type != 'none':
        tp.energy_uv = e_value_grad

    reuse = reuse_struct()
    reuse.fine_recorder = args.fine_recorder

    BC_t = td(BC)
    BN_t = td(BN)
    BCBN = torch.cat([BC_t, BN_t], dim=1)

    value_grad_fun = lambda ddaa: oracle_conjugate_newton_symmetric(
        mesh, ddaa, BCBN, tp, reuse)

    history = []
    num_flipped_old = f
    t0 = time.time()

    u = v = None
    num_flipped = f
    a11 = a12 = a22 = None
    ii = 0

    for ii in range(1, args.max_iter + 1):
        # keep this block the same as core_optimize_block.m
        da, u, v, num_flipped = core_optimize_block(
            ii, da, value_grad_fun, args, f, history, LB, UB, NONLCON)

        if (num_flipped > 0) and (num_flipped <= args.F) \
                and (ii >= args.min_iter_attempt_term) \
                and (num_flipped - num_flipped_old <= 2):

            u, v, num_flipped, a11, a12, a22 = attempt_local_global(
                mesh, da, tp, reuse, BC_t, Area_t, f)   # it computes num_flipped

            if num_flipped == 0 and args.stop_when_no_flip:
                out['finish_with_local_global'] = True
                break

        # stopping_criteria = ((num_flipped<=args.F)&&(num_flipped==0)) ...
        #     || ((num_flipped-num_flipped_old<=0)&&(ii>=100));
        stopping_criteria = (num_flipped <= args.F) and (num_flipped == 0)
        if args.stop_when_no_flip and stopping_criteria:
            break

        num_flipped_old = num_flipped

    t_end = time.time() - t0
    print('***************** Iter %04d, flipps %04d, time: %g********************'
          % (ii, num_flipped, t_end))

    if False:
        au = tp.s_at2au(da.reshape(3, -1).t()).reshape(3, f).t()
        a11_, a12_, a22_ = au[:, 0], au[:, 1], au[:, 2]
        mu = (a22_ - a11_ - 2j * a12_) / ((1 + a11_) * (1 + a22_) - a12_ ** 2)
        uf = torch.abs(mu)   # or torch.angle(mu), mu.imag
        from render_mesh3 import render_mesh3
        render_mesh3(V, F, EdgeColor=[0, 0, 0], FaceScaleColor=uf, ColorMap='default')

    # --- Other possible solvers. ---
    if False:
        # adam
        from fmin_adam import fmin_adam
        sOpt = {'MaxFunEvals': 20, 'MaxIter': 10, 'Display': 'iter'}
        da = fmin_adam(value_grad_fun, da, 0.01, None, None, None, 1, sOpt)

        _, _, _, out_data = value_grad_fun(da)
        u = out_data['u']
        v = out_data['v']

        from render_mesh2 import render_mesh2
        render_mesh2(torch.stack([u, v], dim=1), F,
                     EdgeColor=[0, 0, 0], FaceColor=[1, 1, 1])

        from fmin_vector_adam_simple import fmin_vector_adam_simple
        for _ in range(20):
            # vector adam
            sOpt = {'MaxIter': 50}
            da = fmin_vector_adam_simple(value_grad_fun, da, 0.01, 0.9, 0.999,
                                         sOpt, da.numel() // f)
            _, _, _, out_data = value_grad_fun(da)
            u = out_data['u']
            v = out_data['v']

            flipped = doublearea(np.stack([npy(u), npy(v)], axis=1), F) < 0
            print(int(np.count_nonzero(flipped)))

            render_mesh2(torch.stack([u, v], dim=1), F,
                         EdgeColor=[0, 0, 0], FaceColor=[1, 1, 1])

        # --- A local global solver: Initialization. ---
        import scipy.sparse as sp

        def _row_slice_matrix(indices):
            indices = np.asarray(indices).ravel()
            return sp.coo_matrix((np.ones(indices.size),
                                  (indices, np.arange(indices.size))),
                                 shape=(n, indices.size)).tocsr()

        R = _row_slice_matrix(known)
        S = _row_slice_matrix(unknown)

        au = tp.s_at2au(da.reshape(3, -1).t()).reshape(3, f).t()
        a11 = npy(au[:, 0])
        a12 = npy(au[:, 1])
        a22 = npy(au[:, 2])

        # GI here must be the scipy operator; mesh['GI_sp'] is now torch CSR
        GI = sp.csr_matrix(mesh['GI_sp'].csr.to_dense().cpu().numpy())

        for _ in range(20):
            # global step:
            Lw = GI.T @ sp.bmat([[_sparse_diag(a11), _sparse_diag(a12)],
                                 [_sparse_diag(a12), _sparse_diag(a22)]]) @ GI
            lhs = S.T @ Lw @ S
            rhs = -(S.T @ Lw @ R) @ TVB

            U = sp.linalg.spsolve(sp.csc_matrix(lhs), rhs)

            W = S @ U + R @ TVB

            u = W[:, 0]
            v = W[:, 1]

            print(np.trace(W.T @ (Lw @ W) / 2))

            from render_mesh2 import render_mesh2
            render_mesh2(W, F, EdgeColor=[0, 0, 0])

            # local step:
            if False:
                # the naive slow implementation:
                for j in range(f):
                    Jj = (GI[[j, j + f], :] @ W).T
                    Aj = abs(np.linalg.det(Jj)) * np.linalg.inv(Jj.T @ Jj) * Area[j]
                    a11[j] = Aj[0, 0]
                    a12[j] = Aj[0, 1]
                    a22[j] = Aj[1, 1]
            else:
                # the fast implementation.
                GW = GI @ W
                Gxu = GW[:f, 0]
                Gxv = GW[:f, 1]
                Gyu = GW[f:2 * f, 0]
                Gyv = GW[f:2 * f, 1]

                adetJ = np.abs(Gxu * Gyv - Gxv * Gyu)

                a22 = (Gxu * Gxu + Gxv * Gxv) / adetJ * Area
                a12 = -(Gxu * Gyu + Gxv * Gyv) / adetJ * Area
                a11 = (Gyu * Gyu + Gyv * Gyv) / adetJ * Area

            flipped = doublearea(np.stack([u, v], axis=1), F) < 0
            print(int(np.count_nonzero(flipped)))

        # --- double check ---
        from oracle_conjugate_newton import oracle_conjugate_newton
        _, g1 = oracle_conjugate_newton_symmetric(mesh, da + 0.1, BC_t, tp, reuse)[:2]
        _, g2 = oracle_conjugate_newton(
            mesh, da.reshape(3, f)[[0, 2], :].reshape(-1) + 0.1, BC_t)[:2]
        print(torch.norm(g1.reshape(3, f)[[0, 2], :].reshape(-1) - g2))

        # --- double check intrinsic Laplacian. ---
        from intrinsic_grad import intrinsic_grad
        from cotmatrix import cotmatrix
        GG, _ = intrinsic_grad(V, F)
        LL = GG.T @ _sparse_diag(np.r_[Area, Area]) @ GG
        print(abs(LL + cotmatrix(V, F)).sum() / abs(LL).sum())

    if False:
        from render_mesh3 import render_mesh3
        render_mesh3(torch.stack([u, v], dim=1), F, EdgeColor=[0, 0, 0],
                     FaceScaleColor=flipped, ColorMap='jet')

    rname = args.save_file
    if False:
        np.savez(rname, u=npy(u), v=npy(v), da=npy(da), V=V, F=F)

    out['u'] = u
    out['v'] = v
    out['da'] = da
    out['V'] = V
    out['F'] = F
    out['num_flipped'] = num_flipped
    if args.fine_recorder is not None:
        out['history'] = history
        out['reuse'] = reuse
    out['tp'] = tp
    if False:
        out['mesh'] = mesh
    out['BC'] = BC
    out['BN'] = BN

    out['time'] = t_end

    if out['finish_with_local_global']:
        out['a11'] = a11
        out['a12'] = a12
        out['a22'] = a22

    return out
