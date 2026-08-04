"""FEMoracle.m -- edge-weight oracle: value, gradient in `at` and in the boundary
positions, for the quasi-harmonic map obtained by an inner Laplace solve.

Solver choice: `Lw = LI * diag(au) * LJ` is symmetric only in the
`use_dir_edge == false` branch, and even there `au` carries the *signed* edge
weights of the tensor parameterization, so `Lw(unknown,unknown)` is not SPD --
MATLAB's own comment on line 89 ("note the transpose here for asymmetric Lw!")
makes the asymmetry explicit. Cholesky (ReusableSPDSolver) is therefore invalid;
this port uses `scipy.sparse.linalg.splu` (SuperLU LU, not SuiteSparse/CHOLMOD),
factorizing `Lw_uu` once and reusing it for the transposed solve via
`solve(..., 'T')`.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from tqhm_config import npy
from oracle_joint_hessian import symmetric_tensor_span_dot


def FEMoracle(au, pre, at, BC, s_val_grad, CR):
    verbose = 1
    extra = {}

    use_dir_edge = pre.use_dir_edge

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
    Lfz = pre.DEC['d01']

    if use_dir_edge:
        from triangle_mesh import triangle_mesh
        mesh2 = triangle_mesh(V, F)
        nnz = mesh2['mI'].size
        mI = mesh2['mI'].ravel(order='F')
        mJ = mesh2['mJ'].ravel(order='F')
        LI = sp.coo_matrix((np.ones(nnz), (mI, np.arange(nnz))),
                           shape=(n, nnz)).tocsr()
        LJ = (sp.coo_matrix((np.ones(nnz), (np.arange(nnz), mJ)),
                            shape=(nnz, n)).tocsr()
              - sp.coo_matrix((np.ones(nnz), (np.arange(nnz), mI)),
                              shape=(nnz, n)).tocsr())
    else:
        # Lw = Lfz' * A * Lfz;
        LI = sp.csr_matrix(Lfz.T)
        LJ = sp.csr_matrix(Lfz)

        # A = symmetric_tensor_assemble(au,dim);

    ne = LJ.shape[0]
    au = npy(au).reshape(-1, order='F')
    A = sp.diags(au, 0, shape=(ne, ne), format='csr')
    Lw = (LI @ A @ LJ).tocsr()

    extra['Lw'] = Lw

    Lw_uu = Lw[pre.unknown, :][:, pre.unknown].tocsc()
    Lw_uk = Lw[pre.unknown, :][:, pre.known].tocsr()

    # asymmetric / indefinite -> LU, reused for the transposed solve below
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

    BR = lu.solve(dW[pre.unknown, :], 'T')
    # note the transpose here for asymmetric Lw!

    # PS = Lfz(:,pre.unknown) * BR;
    LI_u = LI[pre.unknown, :]
    PS = np.asarray(LI_u.T @ BR)

    # GW = pre.G(:,pre.unknown)*W(pre.unknown,:) + pre.G(:,pre.known)*W(pre.known,:);
    # GW = Lfz * W;
    GW = np.asarray(LJ @ W)
    for j in range(m):
        if j == 0:
            dEda = np.zeros(ne)

        if True:  # this is much slower.
            spXj = sp.diags(GW[:, j], 0, shape=(ne, ne), format='csr')
            dEda = dEda - spXj.T @ PS[:, j] + CR * 0.5 * (spXj.T @ GW[:, j])
            if j == 0:
                extra['Tx'] = (spXj.T @ LI_u.T).tocsr()
            if j == 1:
                extra['Ty'] = (spXj.T @ LI_u.T).tocsr()
            if j == 2:
                extra['Tz'] = (spXj.T @ LI_u.T).tocsr()
        else:
            dEda = dEda - symmetric_tensor_span_dot(GW[:, j], PS[:, j], dim)

    g = dEda

    extra['g_au'] = g

    # g = pre.s_pdapdt(at)' * g; % too slow
    g = pre.s_pdapdt_lmul(at, g)
    # g = g; % no reparameterization.

    g_BC = dW[pre.known, :] - np.asarray(Lw_uk.T @ lu.solve(dW[pre.unknown, :], 'T'))

    # note the transpose on Lw.

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
