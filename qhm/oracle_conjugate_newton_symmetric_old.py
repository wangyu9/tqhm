"""oracle_conjugate_newton_symmetric_old.m -- the previous generation of
oracle_conjugate_newton_symmetric.m.

The .m file declares `function ... = oracle_conjugate_newton_symmetric(...)`,
i.e. it shadows the newer file under a different name; the Python function is
named after the file, as in tensor_para_old.py.

Differences from the ported new version: u and v carry *different* tensors (A_u
from tp.s_at2au, A_v from tp.s_at2au_v seen through the 90-degree rotation DH),
so the inner solve is real and done twice instead of once in the complex plane;
there is no `reuse` argument, no fixed-sparsity assembly and no boundary term in
the energy; alpha/beta are hard-coded here rather than read from tp.

`symmetric_tensor_span` is not in the MATLAB repo, so `span(xx)' * yy` is
expanded as the `span_dot` that oracle_conjugate_newton_symmetric.m spells out.

The Dirichlet blocks are still solved with ReusableSPDSolver, but the solver is
built per call because this signature carries no `reuse` struct -- that reuse is
exactly what the newer file adds.
"""

import numpy as np
import scipy.sparse as sp
import torch

from tqhm_config import DEV, DT, td, npy
from doublearea import doublearea
from outline import outline
from tdss_solver import ReusableSPDSolver
from res_sym_grad_diag import res_sym_grad_diag
from res_sym_grad_llt import res_sym_grad_llt


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


def _as_fm(at, f):
    """MATLAB reshape(at,[f,numel(at)/f])."""
    if torch.is_tensor(at):
        if at.dim() == 2 and at.shape[0] == f:
            return at
        return at.reshape(-1, f).t().contiguous()
    at = np.asarray(at)
    if at.ndim == 2 and at.shape[0] == f:
        return at
    return at.reshape((f, at.size // f), order='F')


def oracle_conjugate_newton_symmetric_old(mesh, at, BCBN, tp):
    V = npy(mesh['V'])
    F = npy(mesh['F']).astype(np.int64)

    dim = 2

    BCBN = td(BCBN)
    BC = npy(BCBN[:, 0:2])
    BN = npy(BCBN[:, 2:4])

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

    G = mesh['GI_sp'] if 'GI_sp' in mesh else mesh['GI']

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

    if tp['conj_vmap']:
        DH = sp.bmat([[_sparse_diag(np.zeros(f)), -_sparse_diag(np.ones(f))],
                      [_sparse_diag(np.ones(f)), _sparse_diag(np.zeros(f))]],
                     format='csr')
    else:
        DH = sp.eye(2 * f, format='csr')

    u = V[:, 0].copy()
    v = V[:, 1].copy()
    u[known] = BC[:, 0]
    v[known] = BC[:, 1]

    # --- setup parameterizations of A ---

    # [s_at2au,s_at2au_mul,s_pdapdt_lmul] = para_fun_old(Area,dim,'diag');
    # [s_at2au_v,s_at2au_mul_v,s_pdapdt_lmul_v] = para_fun_old(Area,dim,'invdiag');

    para_type = 'diag'   # 'diag-no-mass';

    # [s_at2au,s_pdapdt_lmul,s_at2au_v,s_pdapdt_lmul_v] = para_fun(Area,dim,para_type);

    s_at2au = tp['s_at2au']
    s_pdapdt_lmul = tp['s_pdapdt_lmul']

    if tp['conj_vmap']:
        s_at2au_v = tp['s_at2au_v']
        s_pdapdt_lmul_v = tp['s_pdapdt_lmul_v']
    else:
        s_at2au_v = tp['s_at2au']
        s_pdapdt_lmul_v = tp['s_pdapdt_lmul']

    # ua = da .* [Area;Area];
    # va = 1./da .* [Area;Area];

    at_fm = _as_fm(at, f)

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

    A_u = G.T @ sp.bmat([[_sparse_diag(AA[:, 0, 0]), _sparse_diag(AA[:, 0, 1])],
                         [_sparse_diag(AA[:, 1, 0]), _sparse_diag(AA[:, 1, 1])]],
                        format='csr') @ G
    DHG = DH @ G
    A_v = DHG.T @ sp.bmat([[_sparse_diag(IA[:, 0, 0]), _sparse_diag(IA[:, 0, 1])],
                           [_sparse_diag(IA[:, 1, 0]), _sparse_diag(IA[:, 1, 1])]],
                          format='csr') @ DHG

    A_u = sp.csr_matrix(A_u)
    A_v = sp.csr_matrix(A_v)

    Auu_u = _DirichletSolve(A_u[unknown, :][:, unknown])
    Auu_v = _DirichletSolve(A_v[unknown, :][:, unknown])

    u[unknown] = -Auu_u(A_u[unknown, :][:, known] @ u[known])
    v[unknown] = -Auu_v(A_v[unknown, :][:, known] @ v[known])

    # the span op
    d = 2
    # span = @(xx) symmetric_tensor_span(xx,d);
    span_dot = lambda xx, yy: _span_dot(xx, yy, f)

    Gu = G @ u
    Gv = DHG @ v

    g_u = s_pdapdt_lmul(at_fm, td(
        0.5 * span_dot(Gu, Gu)
        - span_dot(Gu, G @ (S @ Auu_u(S.T @ (A_u @ u))))
    ))

    g_v = s_pdapdt_lmul_v(at_fm, td(
        0.5 * span_dot(Gv, Gv)
        - span_dot(Gv, DHG @ (S @ Auu_v(S.T @ (A_v @ v))))
    ))

    g_bn = 0
    E_bn = 0

    if False:
        assert tp['conj_vmap'] is False

        DiagA = sp.bmat([[_sparse_diag(AA[:, 0, 0]), _sparse_diag(AA[:, 0, 1])],
                         [_sparse_diag(AA[:, 1, 0]), _sparse_diag(AA[:, 1, 1])]],
                        format='csr')

        RE = G @ (A_u @ np.stack([u, v], axis=1) - R @ BN)
        RW = RE - G @ (S @ Auu_u(S.T @ (G.T @ (DiagA @ RE))))

        g_bn = s_pdapdt_lmul(at_fm, td(span_dot(Gu, RW[:, 0]))) \
            + s_pdapdt_lmul(at_fm, td(span_dot(G @ v, RW[:, 1])))

        E_bn = 0.5 * np.sum((A_u @ np.stack([u, v], axis=1) - R @ BN) ** 2)

        # pseudo inverse, PW is unique up to a constant.
        raise NotImplementedError(
            'PW = (G\'*DiagA*G) \\ (R*BN) is a singular all-Neumann Laplacian, '
            'not an SPD system ReusableSPDSolver can factor')

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

    beta = 0
    alpha = 1

    E = alpha * (E_u + E_v) + E_bn * beta

    # render_mesh2([u,v],F,'EdgeColor',[0,0,0]);
    # dda = 1e-7 * normrnd(0,1,size(da));  da = da_old + dda;

    # da = da - 0.001 *  (0.001*speye(numel(da)) + Hess_v) \ (g_u + g_v);
    # da = da - 0.01 * (g_u + g_v);

    g = alpha * (g_u + g_v) + g_bn * beta

    g = g + tp['reg_grad'](at)

    # Hess = Hess_u + Hess_v;
    Hess = None

    out = {}
    out['u'] = td(u)
    out['v'] = td(v)

    return E, g, Hess, out


def _para_fun(FA, dim, para_type):
    """The first local function of oracle_conjugate_newton_symmetric_old.m.

    Dead code there (its only call site is commented out) and superseded by
    tensor_para.m -> tensor_para.py; kept because the `if false` symbolic block
    documents where the res_sym_grad_* closures come from.
    """
    Area = FA
    f = FA.shape[0]

    assert dim == 2

    if para_type == 'diag-no-mass':
        # without mass matrix.
        s_at2au = lambda aatt: torch.cat(
            [aatt[:, 0], torch.zeros(f, dtype=DT, device=aatt.device), aatt[:, 2]])
        s_pdapdt_lmul = lambda aatt, gg: torch.cat(
            [torch.ones(f, dtype=DT, device=gg.device),
             torch.zeros(f, dtype=DT, device=gg.device),
             torch.ones(f, dtype=DT, device=gg.device)]) * gg

        s_at2au_v = lambda aatt: torch.cat(
            [1 / aatt[:, 0], torch.zeros(f, dtype=DT, device=aatt.device), 1 / aatt[:, 2]])
        s_pdapdt_lmul_v = lambda aatt, gg: torch.cat(
            [-1 / aatt[:, 0] ** 2, torch.zeros(f, dtype=DT, device=aatt.device),
             -1 / aatt[:, 2] ** 2]) * gg

    elif para_type == 'diag':
        if False:
            # The syms/matlabFunction derivation of every candidate tensor
            # parameterization; its printed output is what the res_sym_grad_*
            # modules contain.
            raise NotImplementedError(
                'the Symbolic Math Toolbox branch is not ported; the derived '
                'closures live in the res_sym_grad_* modules')
        else:
            # res_sym_grad_diag; res_sym_grad_diag_sq; res_sym_grad_complex_sym;
            sym = res_sym_grad_llt()

        ap = lambda fun: (lambda aatt: fun(aatt[:, 0], aatt[:, 1], aatt[:, 2]))

        s_11, s_12, s_13 = [ap(h) for h in sym['s_dAdP1']]
        s_21, s_22, s_23 = [ap(h) for h in sym['s_dAdP2']]
        s_31, s_32, s_33 = [ap(h) for h in sym['s_dAdP3']]

        t_11, t_12, t_13 = [ap(h) for h in sym['t_dAdP1']]
        t_21, t_22, t_23 = [ap(h) for h in sym['t_dAdP2']]
        t_31, t_32, t_33 = [ap(h) for h in sym['t_dAdP3']]

        if False:
            idf = np.arange(f)
            idfau = np.arange(3 * f)

            def s_pdapdt(aatt):
                A = npy(Area)
                rows = np.concatenate([np.tile(idf + 0 * f, 3),
                                       np.tile(idf + 1 * f, 3),
                                       np.tile(idf + 2 * f, 3)])
                cols = np.tile(idfau, 3)
                vals = np.concatenate([
                    npy(s_11(aatt)) * A, npy(s_12(aatt)) * A, npy(s_13(aatt)) * A,
                    npy(s_21(aatt)) * A, npy(s_22(aatt)) * A, npy(s_23(aatt)) * A,
                    npy(s_31(aatt)) * A, npy(s_32(aatt)) * A, npy(s_33(aatt)) * A,
                ])
                return sp.coo_matrix((vals, (rows, cols)),
                                     shape=(3 * f, idfau.size)).tocsr()

            # not ".*" here, critical to have transpose here.
            s_pdapdt_lmul = lambda aatt, gg: s_pdapdt(aatt).T @ gg
            # this somehows gives incorrect result, check it before use!!!

        s_at2au = lambda aatt: torch.cat([
            Area * ap(sym['mA1'])(aatt),
            Area * ap(sym['mA2'])(aatt),
            Area * ap(sym['mA3'])(aatt),
        ])

        s_pdapdt_lmul = lambda aatt, gg: torch.cat([
            Area * (s_11(aatt) * gg[0:f] + s_21(aatt) * gg[f:2 * f] + s_31(aatt) * gg[2 * f:3 * f]),
            Area * (s_12(aatt) * gg[0:f] + s_22(aatt) * gg[f:2 * f] + s_32(aatt) * gg[2 * f:3 * f]),
            Area * (s_13(aatt) * gg[0:f] + s_23(aatt) * gg[f:2 * f] + s_33(aatt) * gg[2 * f:3 * f]),
        ])

        # this is for diagonal tensor:
        # s_pdapdt_lmul = @(aatt,gg) [Area;zeros(f,1);Area].* gg;

        # s_at2au_v = @(aatt) [Area./aatt(:,1);zeros(f,1);Area./aatt(:,3)];
        # s_pdapdt_lmul_v = @(aatt,gg) [-Area./aatt(:,1).^2;zeros(f,1);-Area./aatt(:,3).^2].* gg;

        s_at2au_v = lambda aatt: torch.cat([
            Area * ap(sym['nA1'])(aatt),
            Area * ap(sym['nA2'])(aatt),
            Area * ap(sym['nA3'])(aatt),
        ])

        s_pdapdt_lmul_v = lambda aatt, gg: torch.cat([
            Area * (t_11(aatt) * gg[0:f] + t_21(aatt) * gg[f:2 * f] + t_31(aatt) * gg[2 * f:3 * f]),
            Area * (t_12(aatt) * gg[0:f] + t_22(aatt) * gg[f:2 * f] + t_32(aatt) * gg[2 * f:3 * f]),
            Area * (t_13(aatt) * gg[0:f] + t_23(aatt) * gg[f:2 * f] + t_33(aatt) * gg[2 * f:3 * f]),
        ])

    elif para_type == 'diag2':
        # with mass matrix.
        # this is for diagonal tensor.
        s_at2au = lambda aatt: torch.cat(
            [Area * aatt[:, 0], torch.zeros(f, dtype=DT, device=aatt.device), Area * aatt[:, 2]])
        s_pdapdt_lmul = lambda aatt, gg: torch.cat(
            [Area, torch.zeros(f, dtype=DT, device=gg.device), Area]) * gg

        s_at2au_v = lambda aatt: torch.cat(
            [Area / aatt[:, 0], torch.zeros(f, dtype=DT, device=aatt.device), Area / aatt[:, 2]])
        s_pdapdt_lmul_v = lambda aatt, gg: torch.cat(
            [-Area / aatt[:, 0] ** 2, torch.zeros(f, dtype=DT, device=gg.device),
             -Area / aatt[:, 2] ** 2]) * gg

    else:
        raise NotImplementedError('Unsupported para type~')

    return s_at2au, s_pdapdt_lmul, s_at2au_v, s_pdapdt_lmul_v


def _para_fun_old(FA, dim, para_type):
    """The second local function of oracle_conjugate_newton_symmetric_old.m.

    Also dead code, and the direct ancestor of tensor_para_old.py. MATLAB derives
    every closure with `syms`; the pre-derived res_sym_grad_* modules stand in,
    which is the same substitution tensor_para.py makes.
    """
    f = FA.shape[0]

    base = np.ones(f)

    idf_cond = np.ones(f)
    idf = np.flatnonzero(idf_cond)
    idn = np.flatnonzero(~idf_cond.astype(bool))
    assert idn.size + idf.size == f
    if dim == 2:
        idfau = np.concatenate([idf, idf + f, idf + 2 * f])
        idnau = np.concatenate([idn, idn + f, idn + 2 * f])
    else:
        assert dim == 3
        idfau = np.concatenate([idf + k * f for k in range(6)])
        idnau = np.concatenate([idn + k * f for k in range(6)])

    if para_type == 'LU':
        delta = 0
        cond_bound = 0
        # aA1 = p1^2 + delta + cond_bound*(p2^2+p3^2); aA2 = p1*p2;
        # aA3 = p2^2 + p3^2 + delta + cond_bound*p1^2
        assert delta == 0 and cond_bound == 0
        sym = res_sym_grad_llt()
        print("LU parameterization of tensor field with delta=%g, cond_bound=%g.\n"
              % (delta, cond_bound), end='')

    elif para_type == 'diag':
        sym = res_sym_grad_diag()

    elif para_type == 'invdiag':
        # aA1 = 1/p1, aA2 = 0, aA3 = 1/p3: the `n`/`t` half of res_sym_grad_diag.
        dg = res_sym_grad_diag()
        sym = {'mA1': dg['nA1'], 'mA2': dg['nA2'], 'mA3': dg['nA3'],
               's_dAdP1': dg['t_dAdP1'], 's_dAdP2': dg['t_dAdP2'],
               's_dAdP3': dg['t_dAdP3']}

    else:
        raise NotImplementedError('Unsupported para type~')

    ap = lambda fun: (lambda aatt: fun(aatt[:, 0], aatt[:, 1], aatt[:, 2]))

    mA1, mA2, mA3 = ap(sym['mA1']), ap(sym['mA2']), ap(sym['mA3'])

    FAf = td(FA)[idf] * td(base)[idf]

    s_at2au = lambda aatt: torch.cat(
        [mA1(aatt) * FAf, mA2(aatt) * FAf, mA3(aatt) * FAf])

    s_at2au_mul_core = s_at2au
    s_at2au_mul = lambda aatt, g: s_at2au_mul_core(aatt)

    # it is critical to use matlabFunction to convert the symbolic function to
    # a function handle.
    s_11, s_12, s_13 = [ap(h) for h in sym['s_dAdP1']]
    s_21, s_22, s_23 = [ap(h) for h in sym['s_dAdP2']]
    s_31, s_32, s_33 = [ap(h) for h in sym['s_dAdP3']]

    # s_pdapdt_old and s_pdapdt are equivalent: s_pdapdt_old(at) - s_pdapdt(at)
    # yields a sparse zero matrix.
    A = npy(FA)[idf]

    def s_pdapdt(aatt):
        rows = np.concatenate([np.tile(idf + 0 * f, 3),
                               np.tile(idf + 1 * f, 3),
                               np.tile(idf + 2 * f, 3)])
        cols = np.tile(idfau, 3)
        vals = np.concatenate([
            npy(s_11(aatt)) * A, npy(s_12(aatt)) * A, npy(s_13(aatt)) * A,
            npy(s_21(aatt)) * A, npy(s_22(aatt)) * A, npy(s_23(aatt)) * A,
            npy(s_31(aatt)) * A, npy(s_32(aatt)) * A, npy(s_33(aatt)) * A,
        ])
        return sp.coo_matrix((vals, (rows, cols)), shape=(3 * f, idfau.size)).tocsr()

    def _sp_mat_T(block):
        return sp.coo_matrix(
            (np.tile(A, 3), (np.tile(idf + block * f, 3), np.arange(idfau.size))),
            shape=(3 * f, idfau.size)).tocsr().T

    sp_mat_T1 = _sp_mat_T(0)
    sp_mat_T2 = _sp_mat_T(1)
    sp_mat_T3 = _sp_mat_T(2)

    # note +0.0*idf is the hack to maintain size when s_ij yields zero.
    def s_pdapdt_lmul(aatt, g):
        gg = npy(g).ravel()
        return (td(sp_mat_T1 @ gg) * torch.cat([s_11(aatt), s_12(aatt), s_13(aatt)])
                + td(sp_mat_T2 @ gg) * torch.cat([s_21(aatt), s_22(aatt), s_23(aatt)])
                + td(sp_mat_T3 @ gg) * torch.cat([s_31(aatt), s_32(aatt), s_33(aatt)]))

    return s_at2au, s_at2au_mul, s_pdapdt_lmul
