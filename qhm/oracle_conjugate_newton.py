"""oracle_conjugate_newton.m -- the scalar (isotropic-conformal) predecessor of
oracle_conjugate_newton_symmetric.m.

The free variable is a single per-face weight `da` of length 2f: `A_u` weights
the gradient by `da` and `A_v` weights the 90-degree-rotated gradient by `1/da`,
so u and v solve two different Dirichlet problems. `core_variational_beltrami.m`
uses this only in its `if false` double-check, feeding it the [1st,3rd] blocks of
the symmetric `da`.

`mesh.G` (extrinsic) is used here, not `mesh.GI`; the double-check needs the two
oracles to use the same operator, and triangle_mesh_basic only builds GI, so this
falls back to GI when G is absent -- for a planar mesh the two agree.

Hess is returned as MATLAB builds it: dense n x n blocks from an explicit `inv`.
That is only usable on the small double-check meshes, so it is computed lazily
and Hess is None unless `want_hess=True`.
"""

import numpy as np
import scipy.sparse as sp
import torch

from tqhm_config import td, npy
from doublearea import doublearea
from outline import outline
from tdss_solver import ReusableSPDSolver


def _sparse_diag(ddd):
    d = np.asarray(ddd, dtype=np.float64).ravel()
    return sp.diags(d, 0, shape=(d.size, d.size), format='csr')


def _row_slice_matrix(indices, n):
    indices = np.asarray(indices).ravel()
    return sp.coo_matrix((np.ones(indices.size), (indices, np.arange(indices.size))),
                         shape=(n, indices.size)).tocsr()


class _DirichletSolve:
    """MATLAB `Muu \\ b` for one fixed SPD block."""

    def __init__(self, M):
        M = sp.csr_matrix(M).sorted_indices()
        self.n = M.shape[0]
        self.solver = ReusableSPDSolver(M.indptr, M.indices, self.n)
        self.data = td(M.data)

    def __call__(self, b):
        b = np.asarray(b)
        if b.ndim == 2:
            return np.stack([self(b[:, k]) for k in range(b.shape[1])], axis=1)
        return npy(self.solver.solve_real(self.data, td(b.ravel())))


def oracle_conjugate_newton(mesh, da, BC, want_hess=False):
    V = npy(mesh['V'])
    F = npy(mesh['F']).astype(np.int64)

    BC = npy(td(BC))

    BE = outline(F)
    B = BE[:, 0]

    n = V.shape[0]
    nb = B.size

    # eq_lhs = [sparse(1:nb, B, 1, nb, n*2); sparse(1:nb, B+n, 1, nb, n*2)]
    # eq_rhs = [VT(B,1); VT(B,2)]
    #
    # if false
    # [V2,F2,~,~] = subdivide_with_constraint(V,F,eq_lhs,eq_rhs,1);
    # [VT2,~,~,~] = subdivide_with_constraint(VT,F,eq_lhs,eq_rhs,1);
    # V = V2; F = F2; VT = VT2;
    # end

    BE = outline(F)
    B = BE[:, 0]

    n = V.shape[0]
    nb = B.size

    para = {}
    # coefficients for constraints
    para['CC'] = 10
    # coefficients for energy
    para['CE'] = 1
    # coefficients for regularizer.
    para['CR'] = 0

    # mesh = triangle_mesh(V,F);
    # mesh2 = triangle_mesh(VT,F);

    # d01 is assigned and never used; mesh['DEC'] only exists when the external
    # DEC toolbox is installed (see triangle_mesh.py), so it is looked up softly.
    d01 = mesh.get('DEC', {}).get('d01')

    G = mesh['G'] if 'G' in mesh else (
        mesh['GI_sp'] if 'GI_sp' in mesh else mesh['GI'])
    G = sp.csr_matrix(G)

    n = mesh['n']
    ne = mesh.get('ne')
    f = mesh['f']
    ft = np.ones((1, f))

    # triangle_mesh.m names these B/UB; triangle_mesh_basic.m, which the solver
    # actually builds, calls the same two lists IKB/IUB.
    known = np.asarray(mesh['B'] if 'B' in mesh else mesh['IKB_np']).ravel()
    unknown = np.asarray(mesh['UB'] if 'UB' in mesh else mesh['IUB_np']).ravel()

    R = _row_slice_matrix(known, n)
    S = _row_slice_matrix(unknown, n)

    Area = doublearea(V, F) / 2.0

    DH = sp.bmat([[_sparse_diag(np.zeros(f)), -_sparse_diag(np.ones(f))],
                  [_sparse_diag(np.ones(f)), _sparse_diag(np.zeros(f))]],
                 format='csr')

    u = V[:, 0].copy()
    v = V[:, 1].copy()
    u[known] = BC[:, 0]
    v[known] = BC[:, 1]

    # ua = da .* [Area;Area];
    # va = 1./da .* [Area;Area];

    da = npy(da).ravel(order='F')

    A_u = sp.csr_matrix(G.T @ _sparse_diag(da) @ G)
    DHG = DH @ G
    A_v = sp.csr_matrix(DHG.T @ _sparse_diag(1.0 / da) @ DHG)

    Auu_u = _DirichletSolve(A_u[unknown, :][:, unknown])
    Auu_v = _DirichletSolve(A_v[unknown, :][:, unknown])

    u[unknown] = -Auu_u(A_u[unknown, :][:, known] @ u[known])
    v[unknown] = -Auu_v(A_v[unknown, :][:, known] @ v[known])

    Gu = G @ u
    DHGv = DHG @ v

    g_u = 0.5 * Gu * Gu - Gu * (G @ (S @ Auu_u(S.T @ (A_u @ u))))

    g_v = -(1.0 / da ** 2) * (
        0.5 * DHGv * DHGv
        - DHGv * (DHG @ (S @ Auu_v(S.T @ (A_v @ v))))
    )

    Hess = None
    if want_hess:
        # MATLAB forms these with an explicit dense inv(); only tractable on the
        # small double-check meshes, hence the flag.
        Suu_v = np.linalg.inv((S.T @ (DHG.T @ (_sparse_diag(1.0 / da) @ (DHG @ S)))).toarray())
        Hess_v = np.diag(DHGv ** 2 / da ** 3) \
            + _sparse_diag(1.0 / da ** 2) @ _sparse_diag(DHGv) @ (
                (DHG @ S) @ Suu_v @ (S.T @ G.T @ DH.T)
            ) @ _sparse_diag(DHGv) @ _sparse_diag(1.0 / da ** 2)

        Suu_u = np.linalg.inv((S.T @ (G.T @ (_sparse_diag(da) @ (G @ S)))).toarray())
        Hess_u = -(_sparse_diag(Gu) @ ((G @ S) @ Suu_u @ (S.T @ G.T)) @ _sparse_diag(Gu))

        Hess = td(Hess_u + Hess_v)

    E_u = 0.5 * float(u @ (A_u @ u))
    E_v = 0.5 * float(v @ (A_v @ v))

    E = E_u + E_v

    # render_mesh2([u,v],F,'EdgeColor',[0,0,0]);
    # dda = 1e-7 * normrnd(0,1,size(da));  da = da_old + dda;

    # da = da - 0.001 *  (0.001*speye(numel(da)) + Hess_v) \ (g_u + g_v);
    # da = da - 0.01 * (g_u + g_v);

    g = td(g_u + g_v)

    out = {}
    out['u'] = td(u)
    out['v'] = td(v)

    return E, g, Hess, out
