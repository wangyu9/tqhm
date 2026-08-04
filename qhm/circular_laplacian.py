"""circular_laplacian.m -- boundary loop operators.

BE: each row holds the two vertex indices of a boundary edge; BE[:,0] is the list
of boundary vertices.
cind: boundary edges in *counterclockwise* order; an index into the rows of BE
(equivalently into B = BE[:,0]).
"""

import numpy as np
import torch

from tqhm_config import DEV, DT, ti, npy
from matlab_set import intersect_stable, intersect_stable_rows


def circular_laplacian(n, F):
    F = ti(F)
    f = int(F.shape[0])
    dcol = F.shape[1]

    # build directed adjacency matrix (copied from outline)
    Fs = F[:, list(range(1, dcol)) + [0]]
    # column-major ravel (MATLAB F(:)) -> transpose then flatten
    I = F.t().reshape(-1)
    J = Fs.t().reshape(-1)
    A = torch.sparse_coo_tensor(torch.stack([I, J]),
                                torch.ones(I.numel(), dtype=DT, device=DEV),
                                size=(n, n)).coalesce()
    D = (A - A.transpose(0, 1)).coalesce()
    Dr, Dc = D.indices()
    Dv = D.values()
    # MATLAB find() is column-major: sort by col (major), row (minor)
    o1 = torch.argsort(Dr, stable=True)
    order = o1[torch.argsort(Dc[o1], stable=True)]
    OI, OJ, OV = Dr[order], Dc[order], Dv[order]
    pos = OV > 0
    BE = torch.stack([OI[pos], OJ[pos]], dim=1)

    BE_np = npy(BE)
    B = BE[:, 0]
    _, _, indE1 = intersect_stable(BE_np[:, 1], BE_np[:, 0])
    _, _, indE2 = intersect_stable(BE_np[:, 0], BE_np[:, 1])
    assert np.array_equal(BE_np[indE1, 0], BE_np[:, 1])
    assert np.array_equal(BE_np[indE2, 1], BE_np[:, 0])

    cind = [0]
    for _ in range(indE1.shape[0] - 1):
        cind.append(indE1[cind[-1]])
    cind = np.array(cind, dtype=np.int64)

    nbe = int(BE.shape[0])
    nbv = nbe                # asserting a simply connected domain

    assert cind.size == nbe  # only works for simply connected meshes

    scind = np.stack([cind, np.r_[cind[1:], cind[0]]], axis=1)

    # for each boundary edge BE[i], BEindF[i] is the triangle it belongs to
    halfedges = np.concatenate([npy(F[:, [0, 1]]), npy(F[:, [1, 2]]),
                                npy(F[:, [2, 0]])], axis=0)
    CC, _, IB = intersect_stable_rows(BE_np, halfedges)
    IF = np.concatenate([np.arange(f)] * 3)
    assert np.array_equal(CC, BE_np)

    BEindF = IF[IB]

    circular_ops = {'BEindF': BEindF}

    nc = int(cind.size)
    eye = torch.sparse_coo_tensor(
        torch.stack([ti(np.arange(nc)), ti(np.arange(nc))]),
        2 * torch.ones(nc, dtype=DT, device=DEV), size=(nc, nc)).coalesce()
    sr = scind.ravel(order='F')
    sc = scind[:, [1, 0]].ravel(order='F')
    L_be = (eye - torch.sparse_coo_tensor(
        torch.stack([ti(sr), ti(sc)]),
        torch.ones(sr.size, dtype=DT, device=DEV),
        size=(nc, nc)).coalesce()).coalesce().to_sparse_csr()

    succ = np.r_[cind[1:], cind[0]]
    prev = np.r_[cind[-1], cind[:-1]]
    # like cind, succ and prev index into the rows of BE and into B

    circular_ops['L_be'] = L_be
    circular_ops['succ'] = succ
    circular_ops['prev'] = prev

    ar = np.arange(nbe)
    Dp = torch.sparse_coo_tensor(torch.stack([ti(prev), ti(ar)]),
                                 torch.ones(nbe, dtype=DT, device=DEV),
                                 size=(nbv, nbe)).coalesce()
    Ds = torch.sparse_coo_tensor(torch.stack([ti(succ), ti(ar)]),
                                 torch.ones(nbe, dtype=DT, device=DEV),
                                 size=(nbv, nbe)).coalesce()

    circular_ops['Dp'] = Dp.to_sparse_csr()
    circular_ops['Ds'] = Ds.to_sparse_csr()

    v2bv = torch.sparse_coo_tensor(torch.stack([ti(ar), B]),
                                   torch.ones(nbv, dtype=DT, device=DEV),
                                   size=(nbv, n)).coalesce()
    # GB computes gradients along the boundary edges
    GB = torch.sparse.mm((Ds - Dp).transpose(0, 1).coalesce(), v2bv)
    circular_ops['GB'] = GB.coalesce().to_sparse_csr()

    Bp = npy(B)[prev]
    Bs = npy(B)[succ]
    Bc = npy(B)[cind]
    Dirac = (torch.sparse_coo_tensor(torch.stack([ti(Bc), ti(Bp)]),
                                     torch.full((nc,), 0.5, dtype=DT, device=DEV),
                                     size=(n, n)).coalesce()
             - torch.sparse_coo_tensor(torch.stack([ti(Bc), ti(Bs)]),
                                       torch.full((nc,), 0.5, dtype=DT, device=DEV),
                                       size=(n, n)).coalesce()).coalesce()
    circular_ops['Dirac'] = Dirac.to_sparse_csr()

    if False:
        # visualization
        import matplotlib.pyplot as plt
        plt.figure()
        BBB = BE[:, 0]
        for i in range(cind.size):
            vv = V[BBB[prev[i]], :]
            plt.scatter(vv[0], vv[1], 10)
            plt.pause(0.1)

    if False:
        import matplotlib.pyplot as plt
        VB = V[B, :]
        dV = V[BE[:, 1], :] - V[BE[:, 0], :]
        plt.quiver(VB[:, 0], VB[:, 1], dV[:, 0], dV[:, 1])

    return BE_np, cind, circular_ops
