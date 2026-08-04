"""upsample_with_weights.m -- Loop-style subdivision without moving the points,
with selective (1-to-4 / 1-to-3 / 1-to-2) refinement.

Yu Wang's adaptation of gptoolbox's `upsample`. This cannot delegate to igl:
igl.upsample only does the uniform 1-to-4 case, has no 'OnlySelected', and
rejects a V with more than 3 columns -- which subdivide_with_constraint.m relies
on (it passes `[V speye(n)]` to read off the prolongation matrix). V may
therefore be a numpy array or a scipy sparse matrix of any width here.

Two notes on the MATLAB source:

- The 4th output `W` is never assigned; the last four lines compute `v1,v2,v3`
  from `V(FF(FO,:))` and drop them (and index V, not VV, so they would be wrong
  anyway). Those lines are preserved and W is returned as None.
- The recursive call is to `upsample`, not to `upsample_with_weights`; since this
  file *is* that algorithm, it recurses into itself here. For `iters == 1` the
  recursion hits `iters < 1` immediately and is a pass-through either way.

The `find(M'==...)` calls walk M row-major (column-major over the transpose),
which is what `np.nonzero(M)` gives, so the face ordering carries over directly.
"""

import numpy as np
import scipy.sparse as sp


def _rows(V, idx):
    return V[np.asarray(idx), :]


def _vstack(parts):
    if any(sp.issparse(p) for p in parts):
        return sp.vstack([sp.csr_matrix(p) for p in parts], format='csr')
    return np.concatenate([np.asarray(p) for p in parts], axis=0)


def _nrows(V):
    return V.shape[0]


def _one_four(offset, V, F):
    """1-to-4 split of every face."""
    E14 = np.concatenate([F[:, [1, 2]], F[:, [2, 0]], F[:, [0, 1]]], axis=0)
    U14 = (_rows(V, E14[:, 0]) + _rows(V, E14[:, 1])) / 2

    nu = E14.shape[0]
    third = nu // 3
    i1 = offset + np.arange(third)
    i2 = offset + third + np.arange(third)
    i3 = offset + 2 * third + np.arange(third)

    # 4 new faces per original face, as if the duplicates in the midpoint list
    # were simply appended to V
    F14 = np.concatenate([
        np.stack([F[:, 0], i3, i2], axis=1),
        np.stack([F[:, 1], i1, i3], axis=1),
        np.stack([F[:, 2], i2, i1], axis=1),
        np.stack([i1, i2, i3], axis=1),
    ], axis=0)
    return U14, F14, E14


def _one_three(offset, V, F, M):
    """1-to-3 split: two of the three edges are subdivided."""
    A = np.cumsum(M, axis=1)

    SI, SJ = np.nonzero(M == 0)
    # vertex opposite the non-subdivided edge
    flip = SJ == 1                      # MATLAB SJ==2, 1-based
    O1 = F[SI, SJ]
    # next vertex
    O2 = F[SI, (SJ + 1) % 3]
    # next next vertex
    O3 = F[SI, (SJ + 2) % 3]

    # vertex opposite the first subdivided edge
    first = (M == 1) & (A == 1)
    SI1, SJ1 = np.nonzero(first)
    E13 = np.stack([F[SI1, (SJ1 + 1) % 3], F[SI1, (SJ1 + 2) % 3]], axis=1)
    # vertex opposite the second subdivided edge
    SI2, SJ2 = np.nonzero((M == 1) & ~first)
    E13 = np.concatenate([
        E13,
        np.stack([F[SI2, (SJ2 + 1) % 3], F[SI2, (SJ2 + 2) % 3]], axis=1),
    ], axis=0)

    # new vertex positions at the midpoints
    U13 = (_rows(V, E13[:, 0]) + _rows(V, E13[:, 1])) / 2

    nu = E13.shape[0]
    half = nu // 2
    i1 = offset + np.arange(half)
    i2 = offset + half + np.arange(half)
    temp1 = i1.copy()
    i1[flip] = i2[flip]
    i2[flip] = temp1[flip]

    F13 = np.concatenate([
        np.stack([i1, O1, i2], axis=1),
        np.stack([i2, O2, O3], axis=1),
        np.stack([i2, O3, i1], axis=1),
    ], axis=0)
    return U13, F13, E13


def _one_two(offset, V, F, M):
    """1-to-2 split: exactly one edge is subdivided."""
    SI, SJ = np.nonzero(M == 1)
    O1 = F[SI, SJ]
    O2 = F[SI, (SJ + 1) % 3]
    O3 = F[SI, (SJ + 2) % 3]
    E12 = np.stack([O2, O3], axis=1)

    U12 = (_rows(V, E12[:, 0]) + _rows(V, E12[:, 1])) / 2

    nu = E12.shape[0]
    i1 = offset + np.arange(nu)
    F12 = np.concatenate([
        np.stack([O1, O2, i1], axis=1),
        np.stack([O1, i1, O3], axis=1),
    ], axis=0)
    return U12, F12, E12


def upsample_with_weights(V, F, KeepDuplicates=False, OnlySelected=None,
                          Iterations=1):
    F = np.asarray(F, dtype=np.int64)

    keep_duplicates = bool(KeepDuplicates)
    sel = OnlySelected
    M = None
    iters = int(Iterations)

    if sel is None or (not callable(sel) and len(np.atleast_1d(sel)) == 0):
        sel = np.arange(F.shape[0])

    if not callable(sel) and np.asarray(sel).shape == F.shape:
        M = np.asarray(sel)
        sel = np.array([], dtype=np.int64)
        assert iters <= 1, 'Specifying M, not compatible with #iterations > 1'

    if iters < 1:
        return V, F, np.arange(F.shape[0]), None

    sel_fun = None
    if callable(sel):
        sel_fun = sel
        sel = sel_fun(V, F)
    sel = np.asarray(sel)
    if sel.dtype == bool:
        sel = np.flatnonzero(sel)
    sel = sel.ravel()

    if F.shape[1] == 3:
        # add a new vertex at the midpoint of each edge
        if M is None:
            # build the unique edge map
            HE = np.sort(F[:, [1, 2, 0, 2, 0, 1]].reshape((-1, 2), order='F'), axis=1)
            uE, EMAP = np.unique(HE, axis=0, return_inverse=True)
            EMAP = EMAP.ravel()
            # all unique edges incident on a selected face
            selmask = np.zeros(F.shape[0], dtype=np.float64)
            selmask[sel] = 1
            uEM = np.bincount(EMAP, weights=np.tile(selmask, 3),
                              minlength=uE.shape[0]) > 0
            # selected half-edges
            M = uEM[EMAP].reshape((-1, 3), order='F')
        else:
            assert M.dtype == bool, 'M should be logical'
            assert M.shape == F.shape, 'M should be #F by 3'

        # for each face, count the half-edges incident on a selected face
        C = M.sum(axis=1)
        # faces touching two selected faces get cut into 3
        M13 = M[C == 2, :]
        S13 = np.flatnonzero(C == 2)
        # these get cut into 2
        M12 = M[C == 1, :]
        S12 = np.flatnonzero(C == 1)
        # even an unselected face gets cut into 4 if all its half-edges split
        S14 = np.flatnonzero(C == 3)
        # faces that are not getting split (*not* the same as the unselected ones)
        S11 = np.setdiff1d(np.arange(F.shape[0]),
                           np.concatenate([S14, S13, S12]))

        n = _nrows(V)
        U14, F14, EU14 = _one_four(n, V, F[S14, :])
        U13, F13, EU13 = _one_three(n + _nrows(U14), V, F[S13, :], M13)
        U12, F12, EU12 = _one_two(n + _nrows(U14) + _nrows(U13), V, F[S12, :], M12)
        F11 = F[S11, :]

        FF = np.concatenate([F14, F13, F12, F11], axis=0)
        FO = np.concatenate([S14, S14, S14, S14, S13, S13, S13, S12, S12, S11])
        U = _vstack([U14, U13, U12])
        EU = np.concatenate([EU14, EU13, EU12], axis=0)
        nu = _nrows(U)

        # find the unique midpoints
        if keep_duplicates:
            J = np.arange(nu)
        else:
            _, I, J = np.unique(np.sort(EU, axis=1), axis=0,
                                return_index=True, return_inverse=True)
            J = J.ravel()
            U = _rows(U, I)
        # append the unique midpoints to the vertex positions
        VV = _vstack([V, U])
        # reindex map from duplicate to unique midpoint indices
        J = np.concatenate([np.arange(n), J + n])
        # reindex faces
        FF = J[FF]

    elif F.shape[1] == 2:
        if sel.size == F.shape[0]:
            m = (_rows(V, F[:, 0]) + _rows(V, F[:, 1])) / 2
            # indices of the new midpoint vertices
            im = _nrows(V) + np.arange(_nrows(m))
            FF = np.concatenate([np.stack([F[:, 0], im], axis=1),
                                 np.stack([im, F[:, 1]], axis=1)], axis=0)
            nf = F.shape[0]
            FO = np.concatenate([np.arange(nf), np.arange(nf)])
            VV = _vstack([V, m])
            # no duplicates in the 2D case
        else:
            Fsel = F[sel, :]
            nsel = np.setdiff1d(np.arange(F.shape[0]), sel)
            Fnsel = F[nsel, :]
            VV, FF, FO, _ = upsample_with_weights(
                V, Fsel, KeepDuplicates=keep_duplicates)
            FF = np.concatenate([Fnsel, FF], axis=0)
            FO = np.concatenate([nsel, sel[FO]])
    else:
        raise ValueError('unsupported simplex size %d' % F.shape[1])

    # recursive call (the iters == 0 base case is handled at the top)
    if sel.size == 0:
        sel = np.arange(F.shape[0])
    if sel_fun is None:
        VV, FF, FOr, _ = upsample_with_weights(
            VV, FF,
            OnlySelected=np.flatnonzero(np.isin(FO, sel)),
            Iterations=iters - 1,
            KeepDuplicates=keep_duplicates)
    else:
        VV, FF, FOr, _ = upsample_with_weights(
            VV, FF,
            OnlySelected=sel_fun,
            Iterations=iters - 1,
            KeepDuplicates=keep_duplicates)
    FO = FO[FOr]

    assert F.shape[1] == 3   # the other case is not implemented

    if False:
        # Dead code in the MATLAB source, and broken: FF indexes into VV, so
        # V(FF(FO,k),:) is out of bounds whenever any vertex was added. Kept
        # because it is what the never-assigned 4th output W was meant to use.
        v1 = _rows(V, FF[FO, 0])
        v2 = _rows(V, FF[FO, 1])
        v3 = _rows(V, FF[FO, 2])

    W = None   # never assigned in the MATLAB source either
    return VV, FF, FO, W
