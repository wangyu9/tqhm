"""oracle_joint_hessian.m -- value, gradient and Hessian of the quasi-harmonic
energy with (u, v, at) as *joint* variables.

Unlike oracle_conjugate_newton_symmetric, the map is not eliminated by an inner
Laplace solve, so the Hessian is a saddle-point block matrix

    [ Auu   0    Fu ]
    [  0   Auu   Fv ]
    [ Fu'  Fv'   H  ]

and is generally indefinite (H is not sign definite and the off-diagonal
coupling is unconstrained). Nothing is solved here -- the matrix is only
assembled -- so no solver choice is made in this file.

`symmetric_tensor_span` / `symmetric_tensor_span_dot` have no .m counterpart
anywhere in ../qhm (MATLAB picks them up off its path); they are inlined here for
d == 2 with the layout [11; 12+21; 22] already used by
oracle_conjugate_newton_symmetric.py.

This is a cold path (no caller in the ported solver), so it works in
numpy/scipy: `g` comes back as a numpy vector and `Hess` as a scipy CSR matrix.

The Hessian block needs `tp.h_dA*_dP*`, which tensor_para.m defines only inside
`if experimental`, so this oracle is unreachable in MATLAB too unless tensor_para
is called with 'Experimental' and the Symbolic Math Toolbox is present. Note also
that `at` here is f-by-2 (third column zeroed), not the f-by-3 used elsewhere --
that is the `assert` on numel(at) in the .m file, not a porting artifact.
"""

import numpy as np
import scipy.sparse as sp
import torch

from tqhm_config import td, npy
from doublearea import doublearea


def _sparse_diag(ddd):
    d = np.asarray(ddd, dtype=np.float64).ravel()
    return sp.diags(d, 0, shape=(d.size, d.size), format='csr')


def symmetric_tensor_span(xx, d):
    """(2f x 3f) matrix with span(x) @ [a11;a12;a22] == DiagA @ x."""
    assert d == 2
    xx = np.asarray(xx, dtype=np.float64).ravel()
    f = xx.size // 2
    Dx = _sparse_diag(xx[:f])
    Dy = _sparse_diag(xx[f:])
    Z = sp.csr_matrix((f, f))
    return sp.bmat([[Dx, Dy, Z], [Z, Dx, Dy]], format='csr')


def symmetric_tensor_span_dot(xx, yy, d):
    """span(xx)' * yy, laid out as [11; 12+21; 22]."""
    assert d == 2
    xx = np.asarray(xx, dtype=np.float64).ravel()
    yy = np.asarray(yy, dtype=np.float64).ravel()
    f = xx.size // 2
    bbx, bby = xx[:f], xx[f:]
    ccx, ccy = yy[:f], yy[f:]
    return np.concatenate([bbx * ccx, bby * ccx + bbx * ccy, bby * ccy])


def _hess_closure(tp, name):
    """tensor_para.m only defines tp.h_dA*_dP* inside `if experimental`."""
    fun = tp[name] if isinstance(tp, dict) else getattr(tp, name, None)
    if fun is None:
        raise NotImplementedError(
            'tp.%s is defined only by the experimental (Symbolic Math Toolbox) '
            'branch of tensor_para.m, which this port does not build' % name)
    return fun


def _as_scipy(GI):
    """mesh.GI is a scipy matrix in triangle_mesh.py but an SpOp elsewhere."""
    return sp.csr_matrix(getattr(GI, 'scipy', GI))


def _row_slice_matrix(indices, n):
    indices = np.asarray(indices, dtype=np.int64).ravel()
    m = indices.size
    return sp.coo_matrix((np.ones(m), (indices, np.arange(m))),
                         shape=(n, m)).tocsr()


def oracle_joint_hessian(mesh, ddaa, BCBN, tp):
    V = npy(mesh['V'])
    F = npy(mesh['F'])

    n = V.shape[0]
    dim = 2

    BCBN = npy(BCBN)
    BC = BCBN[:, 0:2]
    BN = BCBN[:, 2:4]

    known = npy(mesh['IKB']).ravel()
    unknown = npy(mesh['IUB']).ravel()

    nu = unknown.size

    u = np.zeros(n)
    v = np.zeros(n)

    u[known] = BC[:, 0]
    v[known] = BC[:, 1]

    ddaa = npy(ddaa).ravel()
    u[unknown] = ddaa[0:nu]
    v[unknown] = ddaa[nu:2 * nu]
    at = ddaa[2 * nu:]

    f = F.shape[0]

    # if size(at,1)~=f
    #    at = reshape(at,[f,numel(at)/f]);
    # end

    assert at.size == f * 2
    at = at.reshape((f, 2), order='F')
    at = np.concatenate([at, np.zeros((f, 1))], axis=1)

    out = {}

    G = _as_scipy(mesh.get('GI_sp', mesh['GI']))

    R = _row_slice_matrix(known, n)
    S = _row_slice_matrix(unknown, n)

    # the span op
    d = 2
    # this is slow
    span = lambda xx: symmetric_tensor_span(xx, d)
    span_dot = lambda xx, yy: symmetric_tensor_span_dot(xx, yy, d)

    old_imp = False

    s_at2au = tp['s_at2au'] if isinstance(tp, dict) else tp.s_at2au
    s_pdapdt_lmul = tp['s_pdapdt_lmul'] if isinstance(tp, dict) else tp.s_pdapdt_lmul

    at_t = td(at)
    au = npy(s_at2au(at_t)).ravel()

    AA = np.zeros((f, 2, 2))

    a11 = au[0:f]
    a12 = au[f:2 * f]
    a22 = au[2 * f:]

    AA[:, 0, 0] = a11
    AA[:, 0, 1] = a12
    AA[:, 1, 0] = a12
    AA[:, 1, 1] = a22

    DiagA = sp.bmat([[_sparse_diag(AA[:, 0, 0]), _sparse_diag(AA[:, 0, 1])],
                     [_sparse_diag(AA[:, 1, 0]), _sparse_diag(AA[:, 1, 1])]],
                    format='csr')

    A = (G.T @ DiagA @ G).tocsr()

    Auu = A[unknown, :][:, unknown]

    Gx = G[0:f, :]
    Gy = G[f:2 * f, :]

    E_u = 0.5 * (u @ (A @ u)) - 0.5 * (BC[:, 0] @ BN[:, 0])
    E_v = 0.5 * (v @ (A @ v)) - 0.5 * (BC[:, 1] @ BN[:, 1])

    E = E_u + E_v

    Gu = G @ u
    Gv = G @ v

    Gxu = Gx @ u
    Gxv = Gx @ v

    Gyu = Gy @ u
    Gyv = Gy @ v

    hess_terms = {}
    for key in ('h_dA1_dP11', 'h_dA1_dP12', 'h_dA1_dP22',
                'h_dA2_dP11', 'h_dA2_dP12', 'h_dA2_dP22',
                'h_dA3_dP11', 'h_dA3_dP12', 'h_dA3_dP22'):
        hess_terms[key] = npy(_hess_closure(tp, key)(at_t)).ravel()

    h11 = 0.5 * (Gxu ** 2 + Gxv ** 2) * hess_terms['h_dA1_dP11'] \
        + (Gxu * Gyu + Gxv * Gyv) * hess_terms['h_dA2_dP11'] \
        + 0.5 * (Gyu ** 2 + Gyv ** 2) * hess_terms['h_dA3_dP11']

    h12 = 0.5 * (Gxu ** 2 + Gxv ** 2) * hess_terms['h_dA1_dP12'] \
        + (Gxu * Gyu + Gxv * Gyv) * hess_terms['h_dA2_dP12'] \
        + 0.5 * (Gyu ** 2 + Gyv ** 2) * hess_terms['h_dA3_dP12']

    h22 = 0.5 * (Gxu ** 2 + Gxv ** 2) * hess_terms['h_dA1_dP22'] \
        + (Gxu * Gyu + Gxv * Gyv) * hess_terms['h_dA2_dP22'] \
        + 0.5 * (Gyu ** 2 + Gyv ** 2) * hess_terms['h_dA3_dP22']

    H = sp.bmat([[_sparse_diag(h11), _sparse_diag(h12)],
                 [_sparse_diag(h12), _sparse_diag(h22)]], format='csr')

    SS = S
    # or set: SS = sp.eye(n)

    Zss = sp.csr_matrix((SS.shape[1], SS.shape[1]))

    # pdapdt is (2f x 3f)
    pdapdt = sp.csr_matrix(tp['s_pdapdt'](at_t) if isinstance(tp, dict)
                           else tp.s_pdapdt(at_t)).T.tocsr()

    Fu = (SS.T @ G.T @ span(Gu) @ pdapdt.T).tocsr()
    Fv = (SS.T @ G.T @ span(Gv) @ pdapdt.T).tocsr()

    g = np.concatenate([
        SS.T @ (A @ u),
        SS.T @ (A @ v),
        pdapdt @ (0.5 * (span_dot(Gu, Gu) + span_dot(Gv, Gv))),
    ])

    SASS = (SS.T @ A @ SS).tocsr()
    Hess = sp.bmat([[SASS, Zss, Fu],
                    [Zss, SASS, Fv],
                    [Fu.T, Fv.T, H]], format='csr')

    out['u'] = u
    out['v'] = v

    flipped = doublearea(np.stack([u, v], axis=1), F) < 0
    # number of flipped triangles.
    num_flipped = int(np.flatnonzero(flipped).size)
    print('********** flipps %04d\n' % num_flipped, end='')

    return E, g, Hess, out
