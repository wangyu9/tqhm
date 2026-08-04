"""IntrinsicHessianClass.m -- assemble the extrinsic value/gradient/Hessian of a
per-triangle energy written in terms of squared edge lengths.

MATLAB loops over triangles; the `e`/`g`/`h` closures from `intrinsic_grad_hessian`
broadcast, so every loop here is replaced by one batched call. The `idx` variables
that MATLAB computes and never uses inside those loops are dropped.

Layout of the extrinsic gradient/Hessian follows the MATLAB `IDX`, i.e. the 2n
vector is [u; v] (column-major over an n x 2 array), which is the contract
`oracle_conjugate_newton_symmetric` relies on when it slices `r_uv[:n]` and
`r_uv[n:2*n]`.

Note a discrepancy that is preserved from the source: `Value` sums the raw
per-triangle densities while `Grad` weights them by the rest area, so `Grad` is the
gradient of `sum(Area .* e)`, not of `Value`. Same for the WithWeights pair.
"""

import numpy as np
import scipy.sparse as sp
import torch

from tqhm_config import DEV, DT, td, ti
from doublearea import doublearea
from edge_lengths import edge_lengths
from batch_grad_edgesq_over_vertex import batch_grad_edgesq_over_vertex
from extrinsic_edge_hessians import extrinsic_edge_hessians
from grad_edgesq_over_vertex import grad_edgesq_over_vertex


def _edge_lengths_t(V, F):
    """edge_lengths for a torch V; identical ordering (edge i opposite vertex i)."""
    if not torch.is_tensor(V):
        return td(edge_lengths(np.asarray(V), np.asarray(F)))
    F = ti(F) if not torch.is_tensor(F) else F.to(V.device)
    if V.shape[1] == 3:
        V = V[:, 0:3]
    rnr = lambda x: torch.sqrt(torch.sum(x ** 2, dim=1))
    return torch.stack([rnr(V[F[:, 1], :] - V[F[:, 2], :]),
                        rnr(V[F[:, 2], :] - V[F[:, 0], :]),
                        rnr(V[F[:, 0], :] - V[F[:, 1], :])], dim=1)


def _squared_edges(VD, F, V):
    """(a0,a1,b0,b1,c0,c1) as (f,) tensors, in matlabFunction's argument order."""
    E0 = _edge_lengths_t(V, F)
    E = _edge_lengths_t(VD, F)
    return (E0[:, 0] ** 2, E[:, 0] ** 2,
            E0[:, 1] ** 2, E[:, 1] ** 2,
            E0[:, 2] ** 2, E[:, 2] ** 2)


def _idx(F, n):
    """MATLAB IDX: f x 6, columns (x1,y1,x2,y2,x3,y3) into the 2n vector [u;v]."""
    F = np.asarray(F)
    return np.stack([F[:, 0], n + F[:, 0],
                     F[:, 1], n + F[:, 1],
                     F[:, 2], n + F[:, 2]], axis=1)


class IntrinsicHessianClass:

    @staticmethod
    def ComputeIntrinsicGradients(VD, F, V, g):
        """3 x f. VD: deformed pose, V: rest pose."""
        a0, a1, b0, b1, c0, c1 = _squared_edges(VD, F, V)
        return g(a0, a1, b0, b1, c0, c1)

    @staticmethod
    def ComputeIntrinsicHessians(VD, F, V, h):
        """3 x 3 x f."""
        a0, a1, b0, b1, c0, c1 = _squared_edges(VD, F, V)
        return h(a0, a1, b0, b1, c0, c1)

    @staticmethod
    def HessiansIntrinsic2Extrinsic(VD, F, V, Ai, HI, gI, assemble, project):
        """6 x 6 x f from 3 x 3 x f. Ai: area of the rest pose."""
        n = V.shape[0]
        f = np.asarray(F).shape[0]

        VD = td(VD)[:, 0:2]
        Ft = ti(F)
        Ai = td(Ai)

        He1, He2, He3 = extrinsic_edge_hessians()
        He1, He2, He3 = td(He1), td(He2), td(He3)

        IDX = _idx(F, n)

        # KT: deodv, f x 6 x 3
        KT = batch_grad_edgesq_over_vertex(VD, Ft)
        hi = HI.permute(2, 0, 1)                # f x 3 x 3
        KThKT = KT @ hi @ KT.transpose(1, 2)    # f x 6 x 6

        g1, g2, g3 = gI[0, :], gI[1, :], gI[2, :]

        def _eh(x1, x2, x3):
            return (He1[None] * x1[:, None, None]
                    + He2[None] * x2[:, None, None]
                    + He3[None] * x3[:, None, None])

        if project == 0:
            H_ele = (KThKT + _eh(g1, g2, g3)) * Ai[:, None, None]
        elif project == 1:
            Hi = (KThKT + _eh(g1, g2, g3)) * Ai[:, None, None]
            ED0, EV0 = torch.linalg.eigh((Hi + Hi.transpose(1, 2)) / 2)
            H_ele = EV0 @ (ED0.clamp(min=0)[:, :, None] * EV0.transpose(1, 2))
        elif project == 2:
            H_ele = (KThKT + _eh(g1.clamp(min=0), g2.clamp(min=0), g3.clamp(min=0))) \
                * Ai[:, None, None]
        elif project == 3:
            EHi = _eh(g1, g2, g3)
            ED, EV = torch.linalg.eigh(EHi)
            H_ele = (KThKT + EV @ (ED.clamp(min=0)[:, :, None] * EV.transpose(1, 2))) \
                * Ai[:, None, None]
        else:
            raise ValueError('unsupport project')

        # back to MATLAB's 6 x 6 x f page order
        H_ele = H_ele.permute(1, 2, 0).contiguous()

        if assemble:
            EH = H_ele.detach().cpu().numpy()
            rows = np.concatenate([IDX[:, i] for i in range(6) for j in range(6)])
            cols = np.concatenate([IDX[:, j] for i in range(6) for j in range(6)])
            vals = np.concatenate([EH[i, j, :] for i in range(6) for j in range(6)])
            return sp.coo_matrix((vals, (rows, cols)), shape=(n * 2, n * 2)).tocsr()

        return H_ele

    @staticmethod
    def GradsIntrinsic2Extrinsic(VD, F, Area, gI):
        n = VD.shape[0]
        f = np.asarray(F).shape[0]

        dim = np.asarray(F).shape[1] - 1
        assert dim == 2

        VD = td(VD)[:, 0:2]
        Ft = ti(F)
        Area = td(Area).reshape(-1)

        IDX = _idx(F, n)
        IDX_t = ti(IDX)

        if False:   # old code
            gE = torch.zeros(n * 2, dtype=DT, device=DEV)
            for i in range(f):
                VTi = VD[Ft[i, :], :].detach().cpu().numpy()
                KT = td(grad_edgesq_over_vertex(VTi))
                # KT: deodv, 6 x 3
                gE[IDX_t[i, :]] += KT @ gI[:, i] * Area[i]
        else:
            KTs = batch_grad_edgesq_over_vertex(VD, Ft)

            # KTs f x 6 x 3, gI 3 x f, out: f x 6
            AreaKTgI = torch.einsum('fij,jf->fi', KTs, gI) * Area[:, None]

            gE = torch.zeros(n * 2, dtype=DT, device=DEV)
            gE.index_add_(0, IDX_t.reshape(-1), AreaKTgI.reshape(-1))

            if False:
                # MATLAB gates this on `isunix`; it is the same result but much
                # slower, so the vectorized branch is always taken here.
                gE2 = torch.zeros(n * 2, dtype=DT, device=DEV)
                for i in range(f):
                    KT = KTs[i, :, :]
                    gE2[IDX_t[i, :]] += KT @ gI[:, i] * Area[i]
                gE = gE2

        return gE

    @staticmethod
    def AssembleHessian(H_ele, n, F):
        dim = np.asarray(F).shape[1] - 1
        assert dim == 2

        IDX = _idx(F, n)

        # MATLAB never initializes H here (it would error); zero is the intent.
        EH = H_ele.detach().cpu().numpy() if torch.is_tensor(H_ele) else np.asarray(H_ele)
        rows = np.concatenate([IDX[:, i] for i in range(6) for j in range(6)])
        cols = np.concatenate([IDX[:, j] for i in range(6) for j in range(6)])
        vals = np.concatenate([EH[i, j, :] for i in range(6) for j in range(6)])
        return sp.coo_matrix((vals, (rows, cols)), shape=(n * 2, n * 2)).tocsr()

    @staticmethod
    def ValueWithWeights(VD, F, V, ef, W):
        f = np.asarray(F).shape[0]

        W = td(W).reshape(-1)
        assert W.numel() == f

        a0, a1, b0, b1, c0, c1 = _squared_edges(VD, F, V)
        ei = ef(a0, a1, b0, b1, c0, c1)
        return torch.sum(ei * W)

    @staticmethod
    def GradWithWeights(VD, F, V, g, W):
        Area = td(doublearea(np.asarray(V), np.asarray(F)) / 2)

        gI = IntrinsicHessianClass.ComputeIntrinsicGradients(VD, F, V, g)

        return IntrinsicHessianClass.GradsIntrinsic2Extrinsic(
            VD, F, Area * td(W).reshape(-1), gI)

    @staticmethod
    def Value(VD, F, V, ef):
        a0, a1, b0, b1, c0, c1 = _squared_edges(VD, F, V)
        ei = ef(a0, a1, b0, b1, c0, c1)
        return torch.sum(ei)

    @staticmethod
    def Grad(VD, F, V, g):
        Area = td(doublearea(np.asarray(V), np.asarray(F)) / 2)

        gI = IntrinsicHessianClass.ComputeIntrinsicGradients(VD, F, V, g)

        return IntrinsicHessianClass.GradsIntrinsic2Extrinsic(VD, F, Area, gI)

    @staticmethod
    def ProjectedHessian(VD, F, V, g, h):
        # e, g, h = intrinsic_grad_hessian(energy)

        gI = IntrinsicHessianClass.ComputeIntrinsicGradients(VD, F, V, g)
        HI = IntrinsicHessianClass.ComputeIntrinsicHessians(VD, F, V, h)

        dim = np.asarray(F).shape[1] - 1
        assert dim == 2
        Ai = td(doublearea(np.asarray(V), np.asarray(F)) / 2)

        project = 1
        assemble = 1
        return IntrinsicHessianClass.HessiansIntrinsic2Extrinsic(
            VD, F, V, Ai, HI, gI, assemble, project)
