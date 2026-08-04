"""oracle_anisotropic_laplace_joint_grad.m

The joint oracle: (u,v) is a *free* variable here, not the solution of an inner
Dirichlet problem, so this file contains no linear solve. It returns the gradient
of the anisotropic Dirichlet energy with respect to all of (at, u, v) stacked,
plus tp.append_grad for whatever multipliers the caller appended -- an outer
routine is expected to build and solve the joint/KKT system. Nothing here is
SPD-solved, so ReusableSPDSolver is not involved.

`symmetric_tensor_span` is not in the MATLAB repo, so `span(xx)' * yy` is
expanded as the `span_dot` that oracle_conjugate_newton_symmetric.m spells out.
"""

import numpy as np
import scipy.sparse as sp
import torch

from tqhm_config import td, npy
from doublearea import doublearea
from outline import outline


def _sparse_diag(ddd):
    d = np.asarray(ddd, dtype=np.float64).ravel()
    return sp.diags(d, 0, shape=(d.size, d.size), format='csr')


def _row_slice_matrix(indices, n):
    indices = np.asarray(indices).ravel()
    return sp.coo_matrix((np.ones(indices.size), (indices, np.arange(indices.size))),
                         shape=(n, indices.size)).tocsr()


def _span_dot(bb, cc, f):
    """symmetric_tensor_span_dot for d==2: span(bb)' * cc, laid out [11;12+21;22]."""
    bbx, bby = bb[:f], bb[f:2 * f]
    ccx, ccy = cc[:f], cc[f:2 * f]
    return np.concatenate([bbx * ccx, bby * ccx + bbx * ccy, bby * ccy])


def oracle_anisotropic_laplace_joint_grad(mesh, at, uv, BC, tp):
    V = npy(mesh['V'])
    F = npy(mesh['F']).astype(np.int64)

    dim = 2

    BE = outline(F)
    B = BE[:, 0]

    n = V.shape[0]
    nb = B.size

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

    # triangle_mesh.m names these B/UB; triangle_mesh_basic.m calls the same two
    # lists IKB/IUB.
    known = np.asarray(mesh['B'] if 'B' in mesh else mesh['IKB_np']).ravel()
    unknown = np.asarray(mesh['UB'] if 'UB' in mesh else mesh['IUB_np']).ravel()

    R = _row_slice_matrix(known, n)
    S = _row_slice_matrix(unknown, n)

    Area = doublearea(V, F) / 2.0

    # assert(tp.conj_vmap==false);

    if tp['conj_vmap']:
        DH = sp.bmat([[_sparse_diag(np.zeros(f)), -_sparse_diag(np.ones(f))],
                      [_sparse_diag(np.ones(f)), _sparse_diag(np.zeros(f))]],
                     format='csr')
    else:
        DH = sp.eye(2 * f, format='csr')

    uv = npy(uv)
    u = uv[:, 0]
    v = uv[:, 1]

    # --- setup parameterizations of A ---

    # [s_at2au,s_at2au_mul,s_pdapdt_lmul] = para_fun_old(Area,dim,'diag');
    # [s_at2au_v,s_at2au_mul_v,s_pdapdt_lmul_v] = para_fun_old(Area,dim,'invdiag');

    if True:
        para_type = 'diag'   # 'diag-no-mass';

        # [s_at2au,s_pdapdt_lmul,s_at2au_v,s_pdapdt_lmul_v] = para_fun(Area,dim,para_type);

        s_at2au = tp['s_at2au']
        s_pdapdt_lmul = tp['s_pdapdt_lmul']
        s_at2au_v = tp['s_at2au_v']
        s_pdapdt_lmul_v = tp['s_pdapdt_lmul_v']

    # ua = da .* [Area;Area];
    # va = 1./da .* [Area;Area];

    at_t = at if torch.is_tensor(at) else td(at)
    at_fm = at_t if (at_t.dim() == 2 and at_t.shape[0] == f) \
        else at_t.reshape(3, f).t().contiguous()

    au = npy(s_at2au(at_fm))

    AA = np.zeros((f, 2, 2))
    IA = np.zeros((f, 2, 2))

    a11 = au[0:f]
    a12 = au[f:2 * f]
    a22 = au[2 * f:]

    AA[:, 0, 0] = a11
    AA[:, 0, 1] = a12
    AA[:, 1, 0] = a12
    AA[:, 1, 1] = a22

    av = npy(s_at2au_v(at_fm))

    av11 = av[0:f]
    av12 = av[f:2 * f]
    av22 = av[2 * f:]

    IA[:, 0, 0] = av11
    IA[:, 0, 1] = av12
    IA[:, 1, 0] = av12
    IA[:, 1, 1] = av22

    A_u = sp.csr_matrix(
        G.T @ sp.bmat([[_sparse_diag(AA[:, 0, 0]), _sparse_diag(AA[:, 0, 1])],
                       [_sparse_diag(AA[:, 1, 0]), _sparse_diag(AA[:, 1, 1])]],
                      format='csr') @ G)
    DHG = DH @ G
    A_v = sp.csr_matrix(
        DHG.T @ sp.bmat([[_sparse_diag(IA[:, 0, 0]), _sparse_diag(IA[:, 0, 1])],
                         [_sparse_diag(IA[:, 1, 0]), _sparse_diag(IA[:, 1, 1])]],
                        format='csr') @ DHG)

    # the span op
    d = 2
    span_dot = lambda xx, yy: _span_dot(xx, yy, f)

    g_u = A_u @ u

    g_v = A_v @ v

    Gu = G @ u
    DHGv = DHG @ v

    g_at_u = s_pdapdt_lmul(at_fm, td(0.5 * span_dot(Gu, Gu)))

    g_at_v = s_pdapdt_lmul_v(at_fm, td(0.5 * span_dot(DHGv, DHGv)))

    g_at = g_at_u + g_at_v + tp['reg_grad'](at)

    # Hess_v = sparse_diag( (DH * G * v).^2 ./ da.^3 ) ...
    #     + sparse_diag(1./da.^2) * sparse_diag(DH * G * v) * ...
    #     DH * G * S * inv( S' * (DH * G)' * sparse_diag(1./da) * DH * G * S ) * S' * G' * DH' * ...
    #      sparse_diag(DH * G * v) * sparse_diag(1./da.^2);
    #
    # Hess_u = - sparse_diag(G * u) * ...
    #      G * S * inv( S' * (G)' * sparse_diag(da) * G * S ) * S' * G' * ...
    #      sparse_diag( G * u);

    E_u = 0.5 * float(u @ (A_u @ u))
    E_v = 0.5 * float(v @ (A_v @ v))

    E = E_u + E_v

    # render_mesh2([u,v],F,'EdgeColor',[0,0,0]);
    # dda = 1e-7 * normrnd(0,1,size(da));  da = da_old + dda;

    # da = da - 0.001 *  (0.001*speye(numel(da)) + Hess_v) \ (g_u + g_v);
    # da = da - 0.01 * (g_u + g_v);

    # g = [zeros(size(g_at)); g_u; g_v;];
    g = torch.cat([td(g_at), td(g_u), td(g_v)])

    g = torch.cat([g, td(tp['append_grad'])])

    # Hess = Hess_u + Hess_v;
    Hess = None

    out = {}
    out['u'] = td(u)
    out['v'] = td(v)

    return E, g, Hess, out
