"""FEMvalue_ew.m -- the edge-weight variant of FEMoracle.

Differs from FEMoracle.m only in that it always uses `Lfz' * A * Lfz` (no
`use_dir_edge` branch), caches `pre.old_BR`, and solves with `Lw_uu` rather than
its transpose.

Solver choice: `Lw_uu = Lfz(u,:)' diag(au) Lfz(:,u)` is symmetric but `au` holds
the signed edge weights of the tensor parameterization, which may be negative,
so `Lw_uu` is symmetric *indefinite* in general. Cholesky
(ReusableSPDSolver) is therefore invalid; this port uses
`scipy.sparse.linalg.splu` (SuperLU LU, not SuiteSparse/CHOLMOD), factorized
once and reused for both right-hand sides.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from tqhm_config import npy
from oracle_joint_hessian import symmetric_tensor_span_dot


def FEMvalue_ew(au, pre, at, BC, s_val_grad, CR):
    verbose = 1
    extra = {}

    f = pre.f
    dim = pre.dim
    n = pre.n
    solver = 'cholmod'
    BC = np.atleast_2d(npy(BC))
    m = BC.shape[1]
    W = np.zeros((n, m))
    W[pre.known, :] = BC[:, :]
    pre.iter = pre.iter + 1
    print('Iter: %d' % pre.iter, end='')
    nu = pre.unknown.size

    V = pre.V
    F = pre.F

    X = V
    if X.shape[1] == 2:
        X = np.c_[X, np.zeros(X.shape[0])]
    Lfz = sp.csr_matrix(pre.DEC['d01'])

    ne = Lfz.shape[0]

    # A = symmetric_tensor_assemble(au,dim);
    au = npy(au).reshape(-1)
    A = sp.diags(au, 0, shape=(ne, ne), format='csr')

    Lw = (Lfz.T @ A @ Lfz).tocsr()

    extra['Lw'] = Lw

    Lw_uu = Lw[pre.unknown, :][:, pre.unknown].tocsc()
    Lw_uk = Lw[pre.unknown, :][:, pre.known].tocsr()

    # symmetric but indefinite -> LU, not Cholesky
    lu = splu(Lw_uu)
    pre.old_Wu = -lu.solve(np.asarray(Lw_uk @ BC))

    extra['LA'] = Lw_uu
    extra['LB'] = -Lw_uk

    W[pre.unknown, :] = pre.old_Wu

    e, dW = s_val_grad(W)
    e = float(npy(e))

    dW = npy(dW).reshape(W.shape, order='F')

    # dW = dW + CR * 1/2 * Lw * W;

    # e = trace((W'*pre.L)*(pre.invMass*(pre.L*W)));
    # Res = 2 * pre.L*(pre.invMass*(pre.L*W));

    if verbose >= 1:
        print('energy=%f\n' % e, end='')

    pre.old_BR = lu.solve(dW[pre.unknown, :])

    BR = pre.old_BR
    Lfz_u = Lfz[:, pre.unknown]
    PS = np.asarray(Lfz_u @ BR)

    # GW = pre.G(:,pre.unknown)*W(pre.unknown,:) + pre.G(:,pre.known)*W(pre.known,:);
    GW = np.asarray(Lfz @ W)
    for j in range(m):
        if j == 0:
            dEda = np.zeros(ne)

        if True:  # this is much slower.
            spXj = sp.diags(GW[:, j], 0, shape=(ne, ne), format='csr')
            dEda = dEda - spXj.T @ PS[:, j] + CR * 0.5 * (spXj.T @ GW[:, j])
            if j == 0:
                extra['Tx'] = (spXj.T @ Lfz_u).tocsr()
            if j == 1:
                extra['Ty'] = (spXj.T @ Lfz_u).tocsr()
            if j == 2:
                extra['Tz'] = (spXj.T @ Lfz_u).tocsr()
        else:
            dEda = dEda - symmetric_tensor_span_dot(GW[:, j], PS[:, j], dim)

    g = dEda

    extra['g_au'] = g

    # g = pre.s_pdapdt(at)' * g; % too slow
    g = pre.s_pdapdt_lmul(at, g)
    # g = g; % no reparameterization.

    g_BC = dW[pre.known, :] - np.asarray(Lw_uk.T @ lu.solve(dW[pre.unknown, :]))

    # H = speye(size(g,1));
    H = None

    if False:
        folder_path = 'examples/rect-200by40/W_qhe_lbfgs/'
        raise NotImplementedError('the .mat dump of at/g/au is not ported')

    if False:
        from render_mesh2 import render_mesh2
        import matplotlib.pyplot as plt
        render_mesh2(W, pre.F, EdgeColor=[0, 0, 0])
        plt.quiver(W[:, 0], W[:, 1], -dW[:, 0], -dW[:, 1])
        render_mesh2(W, pre.F, EdgeColor=[0, 0, 0])
        plt.quiver(W[pre.known, 0], W[pre.known, 1], -g_BC, -g_BC)

    return e, g, g_BC, W, extra
