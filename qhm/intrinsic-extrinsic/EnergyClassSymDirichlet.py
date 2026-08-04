"""EnergyClassSymDirichlet.m -- symmetric Dirichlet energy, its gradient and the
various quadratic proxies (composite majorization, SLIM, projected Newton, and the
intrinsic/quasi-Newton experiments of the paper).

This is a MATLAB `handle` class with mutable state, so it is ported as a normal
Python class with instance methods of the same names, not as static methods.

Several proxies call into machinery that is not part of this repository:

* `NewtonMex` -- the compiled MEX kernel of the Accelerated Quadratic Proxy /
  Composite Majorization code base (see qhm/compileAllMex.m). Every branch that
  needs it raises NotImplementedError rather than being dropped.
* `ComputeConeHessian` / `ComputeConeHessianLoop` -- also from that code base.

`transposeVec2x2` / `multiplyVec2x2` are the only external helpers simple enough to
reproduce here (per-row 2x2 algebra on an N x 4 array laid out row-major as
[a11, a12, a21, a22]); they are the private `_transpose_vec2x2` /
`_multiply_vec2x2` below.

The matrices D1, D2 and the sparse proxies are f x n / 2n x 2n and this class is
not on the solver's hot path, so it stays in scipy.sparse + numpy, while the
intrinsic-edge routines it delegates to (IntrinsicHessianClass) are torch.

Note two inconsistencies carried over verbatim from the MATLAB. First,
ParameterizerClass.SetMesh calls `Init(..., obj.M.V', obj.M.F')`, i.e. it hands over
the transposed V and F, so `size(obj.V,1)` is 3 rather than the vertex count and
every intrinsic method below that derives `n` from `obj.V` is wrong unless Init was
called with untransposed arrays. Second, `ProxyQuasi` stores its history with
`obj.X_his(nh+1,:,:)` where `nh` is the *current* number of rows, so the array grows
by one row per call and the preallocated `n_his` rows stay zero.
"""

import numpy as np
import scipy.sparse as sp
import torch

from tqhm_config import DEV, DT, td, ti, npy
from edge_lengths import edge_lengths
from intrinsic_grad_hessian import intrinsic_grad_hessian
from IntrinsicHessianClass import IntrinsicHessianClass
from extrinsic_edge_hessians import extrinsic_edge_hessians
from grad_edgesq_over_vertex import grad_edgesq_over_vertex
from grad_edge_over_vertex import grad_edge_over_vertex
from edge_length_per_tri import edge_length_per_tri


def _spdiag(v):
    v = np.asarray(v, dtype=np.float64).ravel()
    return sp.diags(v, 0, shape=(v.size, v.size), format='csr')


def _transpose_vec2x2(M):
    """N x 4 stack of 2x2 matrices [a11,a12,a21,a22] -> their transposes."""
    return np.asarray(M)[:, [0, 2, 1, 3]]


def _multiply_vec2x2(A, B):
    """Per-row 2x2 product of two N x 4 stacks."""
    A = np.asarray(A)
    B = np.asarray(B)
    return np.stack([
        A[:, 0] * B[:, 0] + A[:, 1] * B[:, 2],
        A[:, 0] * B[:, 1] + A[:, 1] * B[:, 3],
        A[:, 2] * B[:, 0] + A[:, 3] * B[:, 2],
        A[:, 2] * B[:, 1] + A[:, 3] * B[:, 3],
    ], axis=1)


def NewtonMex(*args, **kwargs):
    raise NotImplementedError(
        'NewtonMex is a compiled MEX kernel from the AQP/CM code base; not ported')


def ComputeConeHessian(*args, **kwargs):
    raise NotImplementedError(
        'ComputeConeHessian comes from the AQP/CM code base; not ported')


def ComputeConeHessianLoop(*args, **kwargs):
    raise NotImplementedError(
        'ComputeConeHessianLoop comes from the AQP/CM code base; not ported')


class EnergyClassSymDirichlet:

    def __init__(self):
        self.eps = 1e-6          # eps for pushing the PSD proxy to PD

        self.Ai = None           # elements areas
        self.AA = None           # diagonal matrix of element areas repeated 4 times

        self.D1 = None           # derivative operator in frame direction 1
        self.D2 = None           # derivative operator in frame direction 2
        self.DD = None           # nabla operator

        self.x = None            # u,v column stack

        self.fi = None           # energy of terms of the SOS objective
        self.J = None
        self.dS = None
        self.ds = None

        # singular values of J (of the mapping)
        self.S = None
        self.s = None
        self.u1 = None
        self.un = None
        self.v1 = None
        self.vn = None

        # dirichlet proxy (constant w.r.t x)
        self.Hdirichlet = None

        # wangyu added
        self.V = None
        self.F = None

        self.BFGS_iter = 0
        self.BFGS_B = None
        self.BFGS_Bs = None
        self.BFGS_Gs = None
        self.BFGS_x_old = None

        self.H_rest = None

        self.X_his = None
        self.F_his = None
        self.G_his = None
        self.n_his = 10

    def Init(self, D1, D2, Ai, V, F):
        """D1,D2 derivative operators in frame directions, Ai element areas."""
        self.D1 = D1
        self.D2 = D2
        self.Ai = np.asarray(Ai, dtype=np.float64).ravel()
        self.DD = sp.kron(sp.eye(2), sp.vstack([self.D1, self.D2]), format='csr')
        self.AA = _spdiag(np.tile(self.Ai, 4))
        self.Hdirichlet = self.DD.T @ self.AA @ self.DD
        NewtonMex('Init', V, F)   # used in Energy Class

        self.V = V
        self.F = F
        return self

    def ComputeEnergy(self, nargout=2):
        f = float(self.Ai @ self.ComputeEnergyPerElement())
        if nargout > 1:
            g = np.asarray(self.Ai @ self.ComputeGradientPerElement()).ravel()
            return f, g
        return f

    # method used for calculating the scale for min sym-dirichlet energy
    def ComputeSymmetricDirichletParts(self):
        fTerms = self.ComputeEnergyTerms()
        sqrf = fTerms ** 2
        fDirichlet = 0.5 * self.Ai @ (sqrf[:, 0] + sqrf[:, 2])
        finDirichlet = 0.5 * self.Ai @ (sqrf[:, 1] + sqrf[:, 3])
        return fDirichlet, finDirichlet

    def UpdatePose(self, x):
        """x is [u,v] (n x 2) or the column stack [u; v]."""
        x = np.asarray(x, dtype=np.float64)
        self.x = x.ravel(order='F')
        self.SetSingularValues()

    # --- Proxies ---
    def ProxySLIM(self):
        Umat = np.stack([self.u1[:, 0], self.u1[:, 1],
                         self.un[:, 0], self.un[:, 1]], axis=1)
        Umat = _transpose_vec2x2(Umat)
        W = np.zeros((Umat.shape[0], 4))
        W[:, 3] = np.sqrt(0.5 * (self.s - self.s ** -3) / (self.s - 1))
        W[:, 0] = np.sqrt(0.5 * (self.S - self.S ** -3) / (self.S - 1))
        W[np.abs(self.S - 1) < 1e-8, 0] = 1
        W[np.abs(self.s - 1) < 1e-8, 3] = 1
        WW = _multiply_vec2x2(_multiply_vec2x2(Umat, W), _transpose_vec2x2(Umat))

        WWWW = sp.bmat([
            [_spdiag(np.r_[WW[:, 0], WW[:, 0]]), _spdiag(np.r_[WW[:, 1], WW[:, 1]])],
            [_spdiag(np.r_[WW[:, 2], WW[:, 2]]), _spdiag(np.r_[WW[:, 3], WW[:, 3]])],
        ], format='csr')

        WWWDD = WWWW @ self.DD
        return WWWDD.T @ self.AA @ WWWDD

    def ProxyCompMajor(self):
        Js = sp.vstack([self.dS, self.ds], format='csr')
        Hs = np.r_[(1 + 3 * self.S ** -4) * self.Ai, (1 + 3 * self.s ** -4) * self.Ai]
        Hggn = Js.T @ _spdiag(Hs) @ Js   # generalized gauss newton
        gS = self.Ai * (self.S - self.S ** -3)
        gs = self.Ai * (self.s - self.s ** -3)
        walpha = gS + gs
        wbeta = gS - gs
        walpha[walpha < 0] = 0

        # similarity / antisimilarity cone coefficients
        a1 = 0.5 * sp.hstack([self.D1, self.D2], format='csr')
        a2 = 0.5 * sp.hstack([-self.D2, self.D1], format='csr')
        b1 = 0.5 * sp.hstack([self.D1, -self.D2], format='csr')
        b2 = 0.5 * sp.hstack([self.D2, self.D1], format='csr')
        ha = ComputeConeHessian(a1, a2, self.x, walpha)
        hb = ComputeConeHessian(b1, b2, self.x, wbeta)

        H = Hggn + ha + hb

        if False:
            bNoProjection = 1
            H2 = NewtonMex('Compute', self.x, bNoProjection)
            np.linalg.norm(H.toarray().ravel(order='F') - H2.ravel(order='F'))

        return H

    # This is a reference implementation which is less vectorized but might be
    # easier to read
    def ProxyCompMajorLoop(self):
        D1 = self.D1 / 2
        D2 = self.D2 / 2
        a1 = sp.hstack([D1, D2], format='csr')
        a2 = sp.hstack([-D2, D1], format='csr')
        b1 = sp.hstack([D1, -D2], format='csr')
        b2 = sp.hstack([D2, D1], format='csr')
        ha = ComputeConeHessianLoop(a1, a2, self.x)
        hb = ComputeConeHessianLoop(b1, b2, self.x)

        # `obj.Energy` inside this method is a leftover from ParameterizerClass;
        # this class *is* the energy, so the fields are read off self.
        s = self.s
        S = self.S
        self.ComputeSingularDerivatives()
        dS, ds = self.dS, self.ds
        Js = sp.vstack([dS, ds], format='csr')
        Hs = np.r_[(1 + 3 * S ** -4) * self.Ai, (1 + 3 * s ** -4) * self.Ai]
        Hggn = Js.T @ _spdiag(Hs) @ Js   # generalized gauss newton
        gS = self.Ai * (S - S ** -3)
        gs = self.Ai * (s - s ** -3)

        walpha = gS + gs
        wbeta = gS - gs
        walpha[walpha < 0] = 0

        Hab = np.zeros(Hggn.shape)
        for i in range(len(ha)):
            Hab = Hab + walpha[i] * ha[i]
            Hab = Hab + wbeta[i] * hb[i]
        return Hggn + Hab

    def ProxyNewton(self):
        # bNoProjection - flag mitigating performing per face PSD projection
        #   bNoProjection = 0; % means not simple Newton and perform face projection.
        #   default is 0 (perform face projection)
        bNoProjection = 1
        return NewtonMex('Compute', self.x, bNoProjection)

    def ProxyProjectedNewton(self):
        # For Hessians for each face, do this:
        #   [H,HH] = NewtonMex('Compute',x);
        return NewtonMex('Compute', self.x)

    def ProxyTemp(self):
        # return self.ProxyIntrinsic()
        return self.ProxyQuasi()

    def ProxyIntrinsic(self):
        return self.ComputeEdgeHessians(True)

    def ComputeIntrinsicGradients(self):
        n = self.V.shape[0]
        VD = np.asarray(self.x).reshape((n, 2), order='F')
        _, g, h = intrinsic_grad_hessian('symmetric-Dirichlet')
        return IntrinsicHessianClass.ComputeIntrinsicGradients(td(VD), self.F, self.V, g)

    def ComputeIntrinsicHessians(self):
        n = self.V.shape[0]
        VD = np.asarray(self.x).reshape((n, 2), order='F')
        _, g, h = intrinsic_grad_hessian('symmetric-Dirichlet')
        return IntrinsicHessianClass.ComputeIntrinsicHessians(td(VD), self.F, self.V, h)

    def HessiansIntrinsic2Extrinsic(self, HI, gI, assemble, project):
        n = self.V.shape[0]
        VD = np.asarray(self.x).reshape((n, 2), order='F')
        return IntrinsicHessianClass.HessiansIntrinsic2Extrinsic(
            td(VD), self.F, self.V, self.Ai, HI, gI, assemble, project)

    def ComputeIntrinsicGradients2(self):
        """3 x f. Same as ComputeIntrinsicGradients, spelled out."""
        E0 = edge_lengths(self.V, self.F)

        n = self.V.shape[0]
        f = np.asarray(self.F).shape[0]

        VD = np.asarray(self.x).reshape((n, 2), order='F')

        E = edge_lengths(VD, self.F)

        _, g, h = intrinsic_grad_hessian('symmetric-Dirichlet')

        a0 = td(E0[:, 0] ** 2)
        a1 = td(E[:, 0] ** 2)
        b0 = td(E0[:, 1] ** 2)
        b1 = td(E[:, 1] ** 2)
        c0 = td(E0[:, 2] ** 2)
        c1 = td(E[:, 2] ** 2)

        return g(a0, a1, b0, b1, c0, c1)

    def ComputeIntrinsicHessians2(self):
        """3 x 3 x f. Same as ComputeIntrinsicHessians, spelled out."""
        E0 = edge_lengths(self.V, self.F)

        n = self.V.shape[0]
        f = np.asarray(self.F).shape[0]

        VD = np.asarray(self.x).reshape((n, 2), order='F')

        E = edge_lengths(VD, self.F)

        _, g, h = intrinsic_grad_hessian('symmetric-Dirichlet')

        a0 = td(E0[:, 0] ** 2)
        a1 = td(E[:, 0] ** 2)
        b0 = td(E0[:, 1] ** 2)
        b1 = td(E[:, 1] ** 2)
        c0 = td(E0[:, 2] ** 2)
        c1 = td(E[:, 2] ** 2)

        return h(a0, a1, b0, b1, c0, c1)

    def HessiansIntrinsic2Extrinsic2(self, HI, gI, assemble, project):
        """6 x 6 x f from 3 x 3 x f. Same as HessiansIntrinsic2Extrinsic, spelled out."""
        n = self.V.shape[0]
        VD = np.asarray(self.x).reshape((n, 2), order='F')
        return IntrinsicHessianClass.HessiansIntrinsic2Extrinsic(
            td(VD), self.F, self.V, self.Ai, HI, gI, assemble, project)

    def ComputeEdgeHessians(self, assemble):
        """6 x 6 x f."""
        _, Hs = NewtonMex('Compute', self.x)
        ge = self.ComputeGradientPerElement()

        n = self.V.shape[0]
        f = np.asarray(self.F).shape[0]

        VD = np.asarray(self.x).reshape((n, 2), order='F')

        _, g, h = intrinsic_grad_hessian('symmetric-Dirichlet')

        gI = IntrinsicHessianClass.ComputeIntrinsicGradients(td(VD), self.F, self.V, g)
        HI = IntrinsicHessianClass.ComputeIntrinsicHessians(td(VD), self.F, self.V, h)

        # MATLAB computes Hi three times per face and keeps the last assignment,
        # which is the `max(gi,0)` variant, i.e. project == 2.
        project = 2

        if False:
            # verified it agrees with Hs(:,:,i)
            H0 = IntrinsicHessianClass.HessiansIntrinsic2Extrinsic(
                td(VD), self.F, self.V, self.Ai, HI, gI, False, 0)
            assert torch.norm(td(Hs) - H0) <= 1e-8 * torch.norm(td(Hs))

        if False:
            project = 3

        if False:
            project = 1

        H_ele = IntrinsicHessianClass.HessiansIntrinsic2Extrinsic(
            td(VD), self.F, self.V, self.Ai, HI, gI, False, project)

        # gei = full(ge(i,idx))' -- verified it agrees with KT * gi

        if assemble:
            return IntrinsicHessianClass.AssembleHessian(H_ele, n, self.F)
        return H_ele

    def ProxyQuasi(self):
        # For Hessians for each face, do this:
        #   [H,HH] = NewtonMex('Compute',x);
        E0 = edge_lengths(self.V, self.F)

        n = self.V.shape[0]
        f = np.asarray(self.F).shape[0]

        VD = np.asarray(self.x).reshape((n, 2), order='F')

        E = edge_lengths(VD, self.F)

        _, g, h = intrinsic_grad_hessian('symmetric-Dirichlet')

        He1, He2, He3 = extrinsic_edge_hessians()

        F = np.asarray(self.F)
        IDX = np.stack([F[:, 0], n + F[:, 0],
                        F[:, 1], n + F[:, 1],
                        F[:, 2], n + F[:, 2]], axis=1)

        # restart_cond = (self.BFGS_iter % 5 == 0)
        # restart_cond = self.BFGS_iter < 10
        # restart_cond = self.BFGS_iter == 0
        restart_cond = True
        # restart_cond = self.BFGS_iter <= 12

        edge_hess = True
        if edge_hess:
            p = 3
        else:
            p = 6

        if self.F_his is None:
            self.X_his = np.zeros((self.n_his, p, f))
            self.F_his = np.zeros((self.n_his, f))
            self.G_his = np.zeros((self.n_his, p, f))

        if restart_cond:
            # self.Hdirichlet

            self.BFGS_Gs = torch.zeros(p, f, dtype=DT, device=DEV)

            # note the local hessians are always unprojected.
            # as in the c++ code, the 6 by 6 matrix is indexed as
            # x1,y1,x2,y2,x3,y3, indices 1,2,3 corresponds to F(:,1),F(:,2),F(:,3)

            if True:
                if edge_hess:
                    self.BFGS_Bs = self.ComputeIntrinsicHessians()
                    # not area weighted
                else:
                    self.BFGS_B = NewtonMex('Compute', self.x)
                    _, self.BFGS_Bs = NewtonMex('Compute', self.x)
                    # is area weighted
            else:
                self.BFGS_Bs = torch.eye(p, dtype=DT, device=DEV)[:, :, None] \
                    * td(self.Ai)[None, None, :]

        if True:
            if edge_hess:
                gI = self.ComputeIntrinsicGradients()
                EL = edge_length_per_tri(td(VD), ti(self.F))
            else:
                ge = self.ComputeGradientPerElement()
                # ge has not been weighted with per-element area yet

            # the energies are unweighted with area.
            Fs = self.Ai * self.ComputeEnergyPerElement()

            if edge_hess:
                gi = gI                             # p x f, not area weighted
                Xs = EL.t().contiguous()            # p x f
            else:
                gi = td(np.asarray(ge[np.arange(f)[:, None], IDX]).T) * td(self.Ai)[None, :]
                Xs = td(self.x[IDX].T)              # is area weighted

            gi_old = self.BFGS_Gs
            self.BFGS_Gs = gi
            yk = gi - gi_old

            if False:
                # if not restart_cond
                if edge_hess:
                    sk = (EL - td(self.BFGS_x_old)).t()
                else:
                    sk = td(self.x[IDX] - self.BFGS_x_old[IDX]).t()

                Bk = self.BFGS_Bs

                formula = 5

                for i in range(f):
                    Bki = Bk[:, :, i]
                    yki = yk[:, i]
                    ski = sk[:, i]

                    if formula == 0:
                        # BFGS update; note it is not psd for nonconvex
                        # Bkp = Bk + yk*yk'/(yk'*sk) - Bk*sk*(Bk*sk)'/(sk'*Bk*sk)
                        Bkp = Bki
                        if abs(float(yki @ ski)) > 0:
                            Bkp = Bkp + torch.outer(yki, yki) / (yki @ ski)
                        if abs(float(ski @ Bki @ ski)) > 0:
                            Bkp = Bkp - torch.outer(Bki @ ski, Bki @ ski) / (ski @ Bki @ ski)
                    elif formula == 1:
                        # ad-hoc
                        Bkp = Bki + (torch.outer(yki, yki) / (yki @ ski)
                                     - torch.outer(Bki @ ski, Bki @ ski) / (ski @ Bki @ ski)) \
                            * float((yki @ ski) > 0)
                    elif formula == 2:
                        # ad-hoc2
                        Bkp = Bki + torch.outer(yki, yki) / abs(float(yki @ ski)) \
                            - torch.outer(Bki @ ski, Bki @ ski) / (ski @ Bki @ ski)
                    elif formula == 5:
                        # symmetric-rank-one (SR1) formula
                        Bkp = Bki
                        r = yki - Bki @ ski
                        if float(r @ ski) > 0:
                            Bkp = Bki + torch.outer(r, r) / (r @ ski)
                    else:
                        raise ValueError('undefined formula!')

                    if bool(torch.isnan(Bkp).any()):
                        raise ValueError('Bkp has nan')

                    self.BFGS_Bs[:, :, i] = Bkp
            else:
                if True:
                    Bs = self.BFGS_Bs.permute(2, 0, 1)
                    DDD, VVV = torch.linalg.eigh((Bs + Bs.transpose(1, 2)) / 2)
                    # e[e < 0] = 0
                    e = torch.abs(DDD)
                    self.BFGS_Bs = (VVV @ (e[:, :, None] * VVV.transpose(1, 2))) \
                        .permute(1, 2, 0).contiguous()

        nh = self.G_his.shape[0]
        self.X_his = np.concatenate([self.X_his, npy(Xs)[None]], axis=0)
        self.G_his = np.concatenate([self.G_his, npy(self.BFGS_Gs)[None]], axis=0)
        self.F_his = np.concatenate([self.F_his, np.asarray(Fs)[None]], axis=0)

        self.BFGS_iter = self.BFGS_iter + 1

        if edge_hess:
            self.BFGS_x_old = npy(EL)
        else:
            self.BFGS_x_old = self.x

        if edge_hess:
            if False:
                H_ele = torch.zeros(6, 6, f, dtype=DT, device=DEV)
                for i in range(f):
                    VX = self.x[IDX[i, :]].reshape((2, 3), order='F').T
                    KT = td(grad_edge_over_vertex(VX))
                    # KT: deodv, 6 x 3
                    H_ele[:, :, i] = KT @ self.BFGS_Bs[:, :, i] @ KT.T \
                        + 1e-4 * torch.eye(6, dtype=DT, device=DEV)

            project = 1   # 0: almost no progress.
            H_ele = self.HessiansIntrinsic2Extrinsic(self.BFGS_Bs, gI, False, project)
        else:
            H_ele = self.BFGS_Bs

        H = IntrinsicHessianClass.AssembleHessian(H_ele, n, self.F)

        # checked that at iter 1, sum(sum(abs(H - self.BFGS_B))) is ~0
        # H = self.BFGS_B   # should remove this line
        # H = self.BFGS_B * 0.0 + H * 1

        return H

    def ProxyProjectedNewtonFull(self):
        H = self.ProxyNewton()
        H = np.asarray(H.todense()) if sp.issparse(H) else np.asarray(H)
        e, V = np.linalg.eigh(0.5 * (H + H.T))
        e[e < self.eps] = self.eps
        return V @ np.diag(e) @ V.T

    # --- methods(Access = private) ---
    def ComputeEnergyPerElement(self):
        self.fi = self.ComputeEnergyTerms()
        return 0.5 * np.sum(self.fi ** 2, axis=1)

    def ComputeEnergyTerms(self):
        return np.stack([self.S, 1.0 / self.S, self.s, 1.0 / self.s], axis=1)

    def ComputeGradientPerElement(self):
        self.ComputeSingularDerivatives()
        Nf = self.S.size

        dSE = np.stack([np.ones(Nf), -1.0 / self.S ** 2], axis=1)
        dsE = np.stack([np.ones(Nf), -1.0 / self.s ** 2], axis=1)

        gS = dSE[:, 0] * self.fi[:, 0] + dSE[:, 1] * self.fi[:, 1]
        gs = dsE[:, 0] * self.fi[:, 2] + dsE[:, 1] * self.fi[:, 3]

        return _spdiag(gS) @ self.dS + _spdiag(gs) @ self.ds

    def SetSingularValues(self):
        Nv = self.x.size // 2
        U = self.x[:Nv]
        Vx = self.x[Nv:]

        # entries of the Jacobian of all elements (triangles)
        a = self.D1 @ U
        b = self.D2 @ U
        c = self.D1 @ Vx
        d = self.D2 @ Vx

        # save for checking flips of elements
        self.J = a * d - b * c

        UV = np.stack([U, Vx], axis=1)
        D1P = (self.D1 @ UV) / 2
        D2P = (self.D2 @ UV) / 2
        E = D1P[:, 0] + D2P[:, 1]
        Fq = D1P[:, 0] - D2P[:, 1]
        G = D2P[:, 0] + D1P[:, 1]
        H = -D2P[:, 0] + D1P[:, 1]
        Q = np.sqrt(E ** 2 + H ** 2)
        R = np.sqrt(Fq ** 2 + G ** 2)

        a1 = np.arctan2(G, Fq)
        a2 = np.arctan2(H, E)
        theta = (a2 - a1) / 2
        phi = (a2 + a1) / 2

        # save singular values and vectors
        self.s = Q - R
        self.S = Q + R
        self.u1 = np.stack([np.cos(phi), np.sin(phi)], axis=1)
        self.un = np.stack([-np.sin(phi), np.cos(phi)], axis=1)
        self.v1 = np.stack([np.cos(theta), -np.sin(theta)], axis=1)
        self.vn = np.stack([np.sin(theta), np.cos(theta)], axis=1)

    def ComputeSingularDerivatives(self):
        b = _spdiag(self.v1[:, 0]) @ self.D1 + _spdiag(self.v1[:, 1]) @ self.D2
        c = _spdiag(self.vn[:, 0]) @ self.D1 + _spdiag(self.vn[:, 1]) @ self.D2
        self.dS = sp.hstack([_spdiag(self.u1[:, 0]) @ b,
                             _spdiag(self.u1[:, 1]) @ b], format='csr')
        self.ds = sp.hstack([_spdiag(self.un[:, 0]) @ c,
                             _spdiag(self.un[:, 1]) @ c], format='csr')
