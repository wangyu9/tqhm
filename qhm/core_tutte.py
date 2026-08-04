"""core_tutte.m -- a single-shot variant of core_variational_beltrami: evaluate the
oracle once at a tiny random `da` and report the flip count.

MATLAB script operating on the caller's workspace (`folder`, `args`); a function
here, like core_variational_beltrami.py. Deviations, all forced:

- The MATLAB source loads `result.obj` for the target, which the test cases in
  this repo do not ship; `du2020` (target from the OBJ's vt lines) is used when
  it is missing, matching core_variational_beltrami.m's default branch.
- MATLAB calls `oracle_conjugate_newton_symmetric(mesh, ddaa, ..., tp)` with four
  arguments, but that function has taken a fifth `reuse` argument for some time,
  so the script as written no longer runs in MATLAB either. A fresh reuse_struct
  is passed here.
- `mesh.GI = intrinsic_grad_equilateral(n,F)` in the graph_tutte branch drops the
  GIS output the oracle needs; both are set here.
- The `fprintf` at the end prints `ii`, which the script only defines if
  `args.sub_div_level > 0`; it is initialized to 0 here.
"""

import time

import numpy as np
import scipy.sparse as sp
import torch

from tqhm_config import DEV, DT, td, ti, npy
from readOBJ import readOBJ
from upsample import upsample
from doublearea import doublearea
from cotmatrix import cotmatrix
from outline import outline
from triangle_mesh import triangle_mesh
from intrinsic_grad_equilateral import intrinsic_grad_equilateral
from tensor_para import tensor_para
from reuse_struct import reuse_struct
from oracle_conjugate_newton_symmetric import oracle_conjugate_newton_symmetric
from sparse_torch import SpOp


def _sparse_diag(ddd):
    d = np.asarray(ddd, dtype=np.float64).ravel()
    return sp.diags(d, 0, shape=(d.size, d.size), format='csr')


def core_tutte(folder, args):
    V, F, VT, _, _, _ = readOBJ(str(folder) + '/input.obj')
    # [V,F] = readOBJ('bunny_flatten.obj')

    # V = V[:, 0:2]
    VT = VT[:, 0:2]

    ii = 0
    for ii in range(1, args.sub_div_level + 1):
        F_old = F
        V, F = upsample(V, F_old)
        VT, F2 = upsample(VT, F_old)

    mesh = triangle_mesh(V, F)
    mesh2 = triangle_mesh(VT, F)

    # triangle_mesh.m names the boundary/interior lists B/UB and keeps everything
    # in numpy, but oracle_conjugate_newton_symmetric wants IKB/IUB, mesh.AI and
    # torch copies of V/F/GIS (MATLAB's script is stale in exactly the same way:
    # it never sets mesh.IKB/mesh.IUB/mesh.AI either). Bridge it here.
    mesh['IKB_np'] = mesh['B']
    mesh['IUB_np'] = mesh['UB']
    mesh['IKB'] = ti(mesh['B'])
    mesh['IUB'] = ti(mesh['UB'])
    mesh['V_np'] = mesh['V']
    mesh['F_np'] = mesh['F']
    mesh['V'] = td(mesh['V_np'])
    mesh['F'] = ti(mesh['F_np'])
    mesh['GI_sp'] = mesh['GI']
    mesh['GI'] = SpOp(mesh['GI_sp'])
    mesh['GIS_np'] = mesh['GIS']
    mesh['GIS'] = {k: td(v) for k, v in mesh['GIS_np'].items()}

    # input: mesh, VT
    NVT = -cotmatrix(VT, mesh['F_np']) @ VT   # cotmatrix is negative definite

    TVB = VT[mesh['B'], :]
    # target boundary vertices

    # input: V, F, TVB, NVT
    out = {'finish_with_local_global': False}
    # mprint = args.mprint

    tt = args.rotate   # pi/4 * 0.0
    # VT = VT @ [[cos(tt),-sin(tt)],[sin(tt),cos(tt)]].T

    BE = outline(F)
    B = BE[:, 0]

    n = V.shape[0]
    nb = B.size

    # eq_lhs = [sparse(1:nb, B, 1, nb, n*2); sparse(1:nb, B+n, 1, nb, n*2)]
    # eq_rhs = [VT[B,0]; VT[B,1]]

    if False:
        from subdivide_with_constraint import subdivide_with_constraint
        eq_lhs = sp.vstack([
            sp.coo_matrix((np.ones(nb), (np.arange(nb), B)), shape=(nb, 2 * n)),
            sp.coo_matrix((np.ones(nb), (np.arange(nb), B + n)), shape=(nb, 2 * n)),
        ]).tocsr()
        eq_rhs = np.concatenate([VT[B, 0], VT[B, 1]])

        V2, F2, _, _ = subdivide_with_constraint(V, F, eq_lhs, eq_rhs, 1)
        VT2, _, _, _ = subdivide_with_constraint(VT, F, eq_lhs, eq_rhs, 1)

        V = V2
        F = F2
        VT = VT2

    known = mesh['B']
    unknown = mesh['UB']

    def row_slice_matrix(indices):
        indices = np.asarray(indices).ravel()
        return sp.coo_matrix((np.ones(indices.size), (indices, np.arange(indices.size))),
                             shape=(n, indices.size)).tocsr()

    R = row_slice_matrix(known)
    S = row_slice_matrix(unknown)

    BC = TVB
    f = F.shape[0]

    Area = doublearea(V, F) / 2
    mesh['FA'] = None   # to avoid any code using mesh.FA directly

    if args.graph_tutte:
        print('Init with uniform graph Laplacian')
        GI, GIS = intrinsic_grad_equilateral(n, F)
        mesh['GI_sp'] = GI
        mesh['GI'] = SpOp(GI)
        mesh['GIS_np'] = GIS
        mesh['GIS'] = {k: td(v) for k, v in GIS.items()}
        Area = np.ones_like(Area)

    # --- an optional nonlinear energy ---
    # args.energy_type = 'none' / 'ARAP' / 'area' / 'mass-spring' /
    #                    'symmetric-Dirichlet'
    e_value_grad = None
    if args.energy_type != 'none':
        from intrinsic_grad_hessian import intrinsic_grad_hessian
        from IntrinsicHessianClass import IntrinsicHessianClass

        energy_ef, energy_gf, energy_hf = intrinsic_grad_hessian(args.energy_type)

        # s_value_hess_aa = lambda FW: IntrinsicHessianClass.ProjectedHessian(
        #     FW, F, V, energy_gf, energy_hf)

        e_value_grad = lambda FW: (
            IntrinsicHessianClass.Value(FW[:, 0:2], F, V[:, 0:2], energy_ef),
            IntrinsicHessianClass.Grad(FW[:, 0:2], F, V[:, 0:2], energy_gf),
        )

    # --- complex-plane-det1: equivalent ---
    da = np.zeros(3 * f)
    # MATLAB: rng(1,'philox'); Philox is not reproduced here (see PORTING.md).
    g = torch.Generator(device='cpu').manual_seed(1)
    da = ((torch.rand(f, 3, generator=g, dtype=DT) - 0.5) * 0.001).to(DEV)
    LB = None
    UB = None
    NONLCON = None

    Area_t = td(Area)
    mesh['AI'] = Area_t

    tp = tensor_para(Area_t, 2, 'complex-plane-det1')
    tp.conj_vmap = False

    tp.alpha = args.alpha
    tp.beta = args.beta
    tp.gamma = args.gamma

    if args.energy_type != 'none':
        tp.energy_uv = e_value_grad

    BCBN = torch.cat([td(BC), td(R.T @ NVT)], dim=1)
    reuse = reuse_struct()

    value_grad_fun = lambda ddaa: oracle_conjugate_newton_symmetric(
        mesh, ddaa, BCBN, tp, reuse)

    # --- LBFGS ---
    history = []
    num_flipped_old = f
    t0 = time.time()

    _, _, _, out_data = value_grad_fun(da)

    u = out_data['u']
    v = out_data['v']

    flipped = doublearea(np.stack([npy(u), npy(v)], axis=1), F) < 0
    # number of flipped triangles
    num_flipped = int(np.count_nonzero(flipped))

    t_end = time.time() - t0
    print('***************** Iter %04d, flipps %04d, time: %g********************'
          % (ii, num_flipped, t_end))

    out['u'] = u
    out['v'] = v
    out['F'] = F
    out['num_flipped'] = num_flipped
    return out
