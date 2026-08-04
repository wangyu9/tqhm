"""triangle_mesh.m -- the full mesh struct (superset of triangle_mesh_basic).

The `try ... catch` around the DEC experiments is preserved as a try/except that
swallows the failure, matching MATLAB: `getMeshData`/`discreteExteriorCalculus`
come from an external toolbox that is not part of this repo, so that block is
inert here, as it was in the paper's runs.
"""

import numpy as np
import torch

from tqhm_config import DEV, DT, ti
from doublearea import doublearea
from edge_lengths import edge_lengths
from edges import edges
from grad import grad
from mesh_dirac import mesh_dirac
from circular_laplacian import circular_laplacian
from intrinsic_grad import intrinsic_grad
from matlab_set import intersect_stable, intersect_stable_rows


def _coo(vals, rows, cols, shape):
    """Coalesced torch sparse COO on CPU (built once; moved to DEV after mm).

    These boundary-curve operators need sparse x sparse products, and cuSPARSE
    SpGEMM raises "insufficient resources" for larger meshes; the setup runs
    once, so it is assembled on CPU and moved to DEV at the end.
    """
    idx = torch.stack([torch.as_tensor(np.asarray(rows), dtype=torch.int64),
                       torch.as_tensor(np.asarray(cols), dtype=torch.int64)])
    v = torch.as_tensor(np.asarray(vals, dtype=np.float64), dtype=DT)
    return torch.sparse_coo_tensor(idx, v, size=shape).coalesce()


def _basis_grad_core(V, F):
    """Extrinsic gradient of the hat function centered at the triangle's vertex 0."""
    if V.shape[1] == 2:
        V = np.c_[V, np.zeros(V.shape[0])]
    v20 = V[F[:, 2], :] - V[F[:, 0], :]
    v10 = V[F[:, 1], :] - V[F[:, 0], :]
    dot12 = np.sum(v20 * v10, axis=1)
    ns12 = np.sum(np.cross(v20, v10) ** 2, axis=1)
    return (
        v10 * (np.sum(v20 * v20, axis=1) / ns12)[:, None]
        + v20 * (np.sum(v10 * v10, axis=1) / ns12)[:, None]
        - (v10 + v20) * (dot12 / ns12)[:, None]
    )


def _basis_grad_core3d(V, T):
    """Extrinsic gradient of the hat function centered at the tet's vertex 0."""
    assert V.shape[1] == 3
    v30 = V[T[:, 3], :] - V[T[:, 0], :]
    v20 = V[T[:, 2], :] - V[T[:, 0], :]
    v10 = V[T[:, 1], :] - V[T[:, 0], :]
    v31 = v30 - v10
    v21 = v20 - v10
    n123 = np.cross(v21, v31)
    n123 = n123 / np.sqrt(np.sum(n123 ** 2, axis=1))[:, None]
    return n123 * (-1.0 / np.sum(n123 * v10, axis=1))[:, None]


def triangle_mesh(V, F):
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)

    mesh = {}
    n = V.shape[0]
    f = F.shape[0]

    mesh['n'] = n
    mesh['f'] = f

    dim = F.shape[1] - 1
    assert dim in (2, 3)
    mesh['dim'] = dim

    mesh['V'] = V
    mesh['F'] = F

    mesh['D'] = mesh_dirac(n, F)
    mesh['FA'] = doublearea(V, F) / 2.0

    EL = edge_lengths(V, F)
    E = edges(F)
    ne = E.shape[0]

    mesh['EL'] = EL
    mesh['E'] = E
    mesh['ne'] = ne

    assert np.array_equal(np.sort(E, axis=1), E)

    BE, cind, circular_ops = circular_laplacian(n, F)
    mesh['cycle'] = circular_ops
    mesh['cind'] = cind

    B = BE[:, 0]
    SBE = np.sort(BE, axis=1)
    nbe = BE.shape[0]
    nb = nbe                 # asserting a simply connected domain

    mask = np.zeros(n, dtype=bool)
    mask[B] = True
    UB = np.flatnonzero(~mask)

    mesh['B'] = B
    mesh['UB'] = UB
    mesh['BE'] = BE

    _, _, be2e = intersect_stable_rows(SBE, E)
    assert np.array_equal(SBE, E[be2e, :])
    mesh['be2e'] = be2e

    try:
        # BT: list of boundary triangles
        ind_B = np.zeros(n, dtype=np.int64)
        ind_B[B] = 1
        # note: a triangle with only one boundary vertex has no boundary edge
        cnt = ind_B[F[:, 0]] + ind_B[F[:, 1]] + ind_B[F[:, 2]]
        BT = np.flatnonzero(cnt >= 2)

        # assuming each boundary triangle has at most one boundary edge
        assert cnt.max() < 3

        # BTibe: for the k-th triangle in BT, the boundary edge is its BTibe[k]-th
        BTibe = np.argmin(np.stack([ind_B[F[BT, 0]], ind_B[F[BT, 1]], ind_B[F[BT, 2]]], axis=1),
                          axis=1)
        BTvbe = np.zeros((BT.size, 2))
        BTcbe = np.zeros((BT.size, 2))
        for i in range(BT.size):
            a = V[F[BT[i], (BTibe[i] - 1) % 3], 0:2]
            b = V[F[BT[i], (BTibe[i] + 1) % 3], 0:2]
            BTvbe[i, :] = -a + b
            BTcbe[i, :] = (a + b) / 2

        mesh['BT'] = BT
        mesh['BTibe'] = BTibe
        mesh['BTvbe'] = BTvbe
        mesh['BTcbe'] = BTcbe
    except Exception:
        pass

    if False:
        import matplotlib.pyplot as plt
        plt.figure()
        for i in range(BT.size):
            p1 = V[F[BT[i], (BTibe[i] - 1) % 3], 0:2]
            p2 = V[F[BT[i], (BTibe[i] + 1) % 3], 0:2]
            plt.quiver(p1[0], p1[1], p2[0] - p1[0], p2[1] - p1[1])
            plt.pause(0.1)

    if False:
        import matplotlib.pyplot as plt
        plt.figure()
        for i in range(BT.size):
            p1 = V[F[BT[i], (BTibe[i] - 1) % 3], 0:2]
            plt.quiver(p1[0], p1[1], BTvbe[i, 0], BTvbe[i, 1])
            plt.pause(0.1)

    # --- FEM related ---
    g1 = _basis_grad_core(V, F[:, [0, 1, 2]])[:, :2]
    g2 = _basis_grad_core(V, F[:, [1, 2, 0]])[:, :2]
    g3 = _basis_grad_core(V, F[:, [2, 0, 1]])[:, :2]

    mesh['g1'], mesh['g2'], mesh['g3'] = g1, g2, g3

    # --- boundary curve related ---
    ar_be = np.arange(nbe)
    JJJ = _coo(np.ones(nbe), be2e, ar_be, (ne, nbe))
    GB = torch.sparse.mm(JJJ, (
        _coo(np.ones(nbe), ar_be, BE[:, 1], (nbe, n))
        - _coo(np.ones(nbe), ar_be, BE[:, 0], (nbe, n))
    ).coalesce()).coalesce().to_sparse_csr().to(DEV)

    # IBE has the same size as BE but indexes into B, so B[IBE] == BE
    _, _, IBE1 = intersect_stable(BE[:, 0], B)
    _, _, IBE2 = intersect_stable(BE[:, 1], B)
    IBE = np.stack([IBE1, IBE2], axis=1)

    assert np.array_equal(BE, np.stack([B[IBE1], B[IBE2]], axis=1))
    mesh['IBE'] = IBE

    JJJ_T = JJJ.transpose(0, 1).coalesce()
    Dp = torch.sparse.mm(_coo(np.ones(nbe), IBE[:, 0], ar_be, (nb, nbe)),
                         JJJ_T).coalesce().to_sparse_csr().to(DEV)
    Ds = torch.sparse.mm(_coo(np.ones(nbe), IBE[:, 1], ar_be, (nb, nbe)),
                         JJJ_T).coalesce().to_sparse_csr().to(DEV)

    v12 = V[F[:, 1], :] - V[F[:, 0], :]
    v13 = V[F[:, 2], :] - V[F[:, 0], :]

    v23 = V[F[:, 2], :] - V[F[:, 1], :]
    v21 = V[F[:, 0], :] - V[F[:, 1], :]

    v31 = V[F[:, 0], :] - V[F[:, 2], :]
    v32 = V[F[:, 1], :] - V[F[:, 2], :]

    dot23 = np.sum(v12 * v13, axis=1)
    dot31 = np.sum(v23 * v21, axis=1)
    dot12 = np.sum(v31 * v32, axis=1)

    mesh['dot23'] = dot23
    mesh['dot31'] = dot31
    mesh['dot12'] = dot12

    # 2D cross
    cross23 = v12[:, 0] * v13[:, 1] - v12[:, 1] * v13[:, 0]
    cross31 = v23[:, 0] * v21[:, 1] - v23[:, 1] * v21[:, 0]
    cross12 = v31[:, 0] * v32[:, 1] - v31[:, 1] * v32[:, 0]

    mesh['cross23'] = cross23
    mesh['cross31'] = cross31
    mesh['cross12'] = cross12

    # signed tan of the half angle: tan(t/2) = sin(t)/(1+cos(t))
    stan1 = cross23 / (dot23 + EL[:, 1] * EL[:, 2])
    stan2 = cross31 / (dot31 + EL[:, 2] * EL[:, 0])
    stan3 = cross12 / (dot12 + EL[:, 0] * EL[:, 1])

    stan = np.stack([stan1, stan2, stan3], axis=1)
    mesh['stan'] = stan

    mV = np.concatenate([stan / EL[:, [2, 0, 1]], stan / EL[:, [1, 2, 0]]], axis=1)
    mI = np.concatenate([F, F], axis=1)
    mJ = np.concatenate([F[:, [1, 2, 0]], F[:, [2, 0, 1]]], axis=1)

    mesh['WDE'] = mV
    mesh['mI'] = mI
    mesh['mJ'] = mJ

    def assemble_MVL(II, JJ, VV):
        II = np.asarray(II).ravel(order='F')
        JJ = np.asarray(JJ).ravel(order='F')
        VV = np.asarray(VV).ravel(order='F')
        return (_coo(VV, II, JJ, (n, n))
                - _coo(VV, II, II, (n, n))).coalesce().to_sparse_csr().to(DEV)

    mesh['assemble_MVL'] = assemble_MVL
    mesh['MVL'] = assemble_MVL(mI, mJ, mV)

    # WDE is an f x 3 array holding data for each directed edge
    mesh['wde2vec'] = lambda WDE: np.asarray(WDE).ravel(order='F')
    mesh['vec2wde'] = lambda aa: np.asarray(aa).reshape((f, 3), order='F')

    X = V
    if X.shape[1] == 2:
        X = np.c_[X, np.zeros(X.shape[0])]

    try:
        # some experiments that are not used in the paper
        from getMeshData import getMeshData
        from discreteExteriorCalculus import discreteExteriorCalculus
        meshdec = getMeshData(X, F, 2, 'none')
        DEC = discreteExteriorCalculus(meshdec)
        assert np.array_equal(E, meshdec['Elist'])
        mesh['DEC'] = DEC
    except Exception:
        pass

    mesh['G'] = grad(V, F)

    mesh['GI'], mesh['GIS'] = intrinsic_grad(V, F)
    return mesh
