"""core_local_global.m -- a pure local/global driver (no outer tensor solve).

MATLAB script on the caller's workspace (`folder`, `args`); here a function
returning `out`.

Two things are stale in the MATLAB source and are preserved rather than fixed:

* the script never creates `reuse`, so the `old_imp == false` branch of
  `attempt_local_global` -- the only live one -- reads an undefined variable.
  A fresh `reuse_struct()` is passed here so the call is at least well formed;
  it still fails at call time because the sparsity/factorization cache is only
  populated by `oracle_conjugate_newton_symmetric`.
* `cotmatrix` is not part of the gptoolbox subset mirrored in this repo, so the
  `NVT` line raises ImportError at call time.

`triangle_mesh` names the boundary sets `B`/`UB` and keeps `V`/`GIS` in numpy,
while `attempt_local_global` expects the `triangle_mesh_basic` layout
(`IKB`/`IUB`, torch); the adapter dict below bridges the two without changing
which vertices are known.
"""

import numpy as np
import scipy.sparse as sp
import torch

from tqhm_config import td, ti, npy
from readOBJ import readOBJ
from upsample import upsample
from doublearea import doublearea
from triangle_mesh import triangle_mesh
from intrinsic_grad_equilateral import intrinsic_grad_equilateral
from tensor_para import tensor_para
from reuse_struct import reuse_struct
from oracle_conjugate_newton_symmetric import oracle_conjugate_newton_symmetric
from attempt_local_global import attempt_local_global


def _sparse_diag(ddd):
    d = np.asarray(ddd, dtype=np.float64).ravel()
    return sp.diags(d, 0, shape=(d.size, d.size), format='csr')


def core_local_global(folder, args):
    V, F, _, _, _, _ = readOBJ(str(folder) + 'input.obj')
    # V, F = readOBJ('bunny_flatten.obj')

    # V = V[:, 0:2]
    VT, _, _, _, _, _ = readOBJ(str(folder) + 'result.obj')
    # VT = V
    VT = VT[:, 0:2]

    for _ in range(args.sub_div_level):
        F_old = F
        V, F = upsample(V, F_old)
        VT, F2 = upsample(VT, F_old)

    mesh = triangle_mesh(V, F)
    mesh2 = triangle_mesh(VT, F)

    # input: mesh, VT
    from cotmatrix import cotmatrix
    NVT = -(cotmatrix(VT, mesh['F']) @ VT)   # cotmatrix is negative definite.

    TVB = VT[mesh['B'], :]
    # target boundary vertices.

    # input: V, F, TVB, NVT
    out = {'finish_with_local_global': False}
    # mprint = args.mprint

    # Example 2

    tt = args.rotate   # pi/4 * 0.0
    # VT = VT @ [[cos(tt),-sin(tt)],[sin(tt),cos(tt)]].T

    from outline import outline
    BE = outline(F)
    B = BE[:, 0]

    n = V.shape[0]
    nb = B.size

    # eq_lhs = [[sparse(0:nb, B, 1, nb, n*2)], [sparse(0:nb, B+n, 1, nb, n*2)]]
    # eq_rhs = [VT[B,0]; VT[B,1]]

    if False:
        # V2, F2, _, _ = subdivide_with_constraint(V, F, eq_lhs, eq_rhs, 1)
        # VT2, _, _, _ = subdivide_with_constraint(VT, F, eq_lhs, eq_rhs, 1)
        # V = V2
        # F = F2
        # VT = VT2
        pass

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

    Area = doublearea(V, F) / 2.0
    mesh['FA'] = None   # to avoid any code to use mesh.FA directly.

    if args.graph_tutte:
        print('Init with uniform graph Laplacian')
        GI, GIS = intrinsic_grad_equilateral(n, F)
        mesh['GI'] = GI
        mesh['GIS'] = GIS
        Area = np.ones_like(Area)

    # an nonlinear energy

    # args.energy_type = 'none'
    # args.energy_type = 'ARAP'   # 'area', 'mass-spring'; ARAP
    # 'symmetric-Dirichlet'
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
    da = torch.zeros(3 * f, dtype=torch.float64)
    # MATLAB: rng(1,'philox'); Philox is not reproduced here (see PORTING.md).
    g = torch.Generator(device='cpu').manual_seed(1)
    da = td((torch.rand(f, 3, generator=g, dtype=torch.float64) - 0.5) * 0.001)
    LB = None
    UB = None
    NONLCON = None

    Area_t = td(Area)
    tp = tensor_para(Area_t, 2, 'complex-plane-det1')
    tp.conj_vmap = False

    tp.alpha = args.alpha
    tp.beta = args.beta
    tp.gamma = args.gamma

    if args.energy_type != 'none':
        tp.energy_uv = e_value_grad

    reuse = reuse_struct()

    BCBN = torch.cat([td(BC), td(R.T @ NVT)], dim=1)
    value_grad_fun = lambda ddaa: oracle_conjugate_newton_symmetric(
        mesh, ddaa, BCBN, tp, reuse)

    # --- LBFGS ---
    history = []
    num_flipped_old = f

    import time
    t0 = time.time()

    # attempt_local_global wants the triangle_mesh_basic field names.
    lg_mesh = dict(mesh)
    lg_mesh['V'] = td(mesh['V'])
    lg_mesh['IKB'] = ti(known)
    lg_mesh['IUB'] = ti(unknown)
    lg_mesh['GIS'] = {k: td(v) for k, v in mesh['GIS'].items()}

    u = v = None
    num_flipped = f
    ii = 0

    for ii in range(1, args.max_iter + 1):
        if True:
            u, v, num_flipped, a11, a12, a22 = attempt_local_global(
                lg_mesh, da, tp, reuse, td(BC), Area_t, f)

            flipped = doublearea(np.stack([npy(u), npy(v)], axis=1), F) < 0
            # number of flipped triangles.
            num_flipped = int(np.count_nonzero(flipped))
            if num_flipped == 0:
                out['finish_with_local_global'] = True
                break

        stopping_criteria = (num_flipped == 0) \
            or ((num_flipped - num_flipped_old <= 0) and (ii >= 100))
        if stopping_criteria:
            break

        num_flipped_old = num_flipped

    t_end = time.time() - t0
    print('***************** Iter %04d, flipps %04d, time: %g********************'
          % (ii, num_flipped, t_end))

    out['u'] = u
    out['v'] = v
    out['V'] = V
    out['F'] = F
    out['num_flipped'] = num_flipped
    out['time'] = t_end
    return out
