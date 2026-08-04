"""FEMpre.m -- the precomputation object shared by FEMoracle / FEMvalue_ew.

MATLAB's `classdef ... < handle` becomes a plain class: the callers mutate
`pre.iter`, `pre.old_Wu` and `pre.old_BR` in place, which a handle class allows
and a Python object does too.

Two external dependencies are not part of `../qhm`:

* `my_grad` -- absent everywhere in the MATLAB tree (it is on the author's
  path). `pre.G` is only referenced from commented-out lines in FEMoracle.m /
  FEMvalue_ew.m, so gptoolbox's `grad` is used for it.
* `getMeshData` / `discreteExteriorCalculus` -- an external DEC toolbox that
  triangle_mesh.py already treats as absent (its use there sits in a
  try/catch). Only `DEC.d01` and `DEC.star11D` are consumed downstream, and
  both are unambiguous, so `dec_d01` / `dec_star11D` below build them directly
  when the toolbox cannot be imported. The edge ordering is gptoolbox `edges`,
  which is what triangle_mesh.m asserts `DEC.mesh.Elist` equals.
"""

import numpy as np
import scipy.sparse as sp

from doublearea import doublearea
from edges import edges
from edge_lengths import edge_lengths
from grad import grad
from cotmatrix import cotmatrix
from massmatrix import massmatrix
from vertex_facet_adjacency import vertex_facet_adjacency


def dec_d01(V, F):
    """DEC.d01: the ne x n signed edge/vertex incidence (exterior derivative)."""
    n = np.asarray(V).shape[0]
    E = edges(F)
    ne = E.shape[0]
    rows = np.concatenate([np.arange(ne), np.arange(ne)])
    cols = np.concatenate([E[:, 0], E[:, 1]])
    vals = np.concatenate([-np.ones(ne), np.ones(ne)])
    return sp.coo_matrix((vals, (rows, cols)), shape=(ne, n)).tocsr(), E


def dec_star11D(V, F, E=None):
    """DEC.star11D: the diagonal Hodge star on 1-forms, i.e. the cotan weights."""
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F)
    if E is None:
        E = edges(F)

    l2 = edge_lengths(V, F) ** 2
    dblA = doublearea(V, F)
    cot = np.stack([
        l2[:, 1] + l2[:, 2] - l2[:, 0],
        l2[:, 2] + l2[:, 0] - l2[:, 1],
        l2[:, 0] + l2[:, 1] - l2[:, 2],
    ], axis=1) / (2.0 * dblA[:, None])

    # corner k's cotangent weights the opposite edge
    opp = np.concatenate([F[:, [1, 2]], F[:, [2, 0]], F[:, [0, 1]]], axis=0)
    opp = np.sort(opp, axis=1)
    w = 0.5 * np.concatenate([cot[:, 0], cot[:, 1], cot[:, 2]])

    key = E[:, 0].astype(np.int64) * (V.shape[0] + 1) + E[:, 1]
    okey = opp[:, 0].astype(np.int64) * (V.shape[0] + 1) + opp[:, 1]
    idx = np.searchsorted(key, okey)

    diag = np.zeros(E.shape[0])
    np.add.at(diag, idx, w)
    return sp.diags(diag, 0, format='csr')


class FEMpre:
    def __init__(self, V, F, known, BC, s_pdapdt_lmul):
        pre = self

        pre.use_dir_edge = False

        F = np.asarray(F)
        pre.dim = F.shape[1] - 1

        pre.V = np.asarray(V, dtype=np.float64)[:, 0:pre.dim]

        pre.F = F
        pre.known = np.asarray(known, dtype=np.int64).ravel()
        pre.BC = None if BC is None else np.asarray(BC, dtype=np.float64)

        pre.f = pre.F.shape[0]
        pre.G = grad(pre.V, pre.F)

        if pre.dim == 2:
            pre.FA = doublearea(pre.V, pre.F) / 2.0
        else:
            assert pre.dim == 3
            raise NotImplementedError('volume() for tets is not ported')
        if pre.FA.min() < 0:
            print('Warning: The area FA has negative terms, made positive.\n', end='')
            print('%d elements have negative volume.\n'
                  % np.flatnonzero(pre.FA < 0).size, end='')
            pre.FA = np.abs(pre.FA)
        pre.MF = sp.diags(pre.FA, 0, shape=(pre.f, pre.f), format='csr')
        pre.J = vertex_facet_adjacency(pre.F)
        pre.v2f = sp.csr_matrix(
            pre.J.multiply(1.0 / np.asarray(pre.J.sum(axis=1))))
        pre.n = pre.V.shape[0]
        mask = np.zeros(pre.n, dtype=bool)
        mask[pre.known] = True
        pre.unknown = np.flatnonzero(~mask)

        if pre.BC is None or pre.BC.size == 0:
            pre.BC = np.eye(pre.known.size)

        pre.L = -cotmatrix(pre.V, pre.F)
        pre.Mass = massmatrix(pre.V, pre.F, 'voronoi')
        if pre.Mass.diagonal().min() < 0:
            print('Warning: The area Mass has negative terms, made positive.\n', end='')
            print('%d vertices have negative volume.\n'
                  % np.flatnonzero(pre.Mass.diagonal() < 0).size, end='')
            pre.Mass = abs(pre.Mass)
        assert pre.Mass.nnz == pre.n
        pre.invMass = sp.diags(1.0 / pre.Mass.diagonal(), 0, format='csr')

        f = pre.f
        # assert(pre.dim==2);
        # pre.aG = [pre.G(f+1:2*f,:);pre.G(1:f,:)];
        pre.aG = None

        # --- only for the DEC and edge-based approach ---
        if pre.F.shape[1] == 3:
            X = pre.V
            if X.shape[1] == 2:
                X = np.c_[X, np.zeros(X.shape[0])]
            pre.DEC = {}
            pre.DEC['d01'], E = dec_d01(X, pre.F)
            pre.DEC['star11D'] = dec_star11D(X, pre.F, E)
            pre.DEC['mesh'] = {'Elist': E}
            pre.D = pre.DEC['d01']
            pre.star = pre.DEC['star11D']
        else:
            pre.DEC = None
            pre.D = None
            pre.star = None

        pre.old_BR = None
        pre.old_Wu = None

        pre.s_pdapdt_lmul = s_pdapdt_lmul

        pre.iter = 0
