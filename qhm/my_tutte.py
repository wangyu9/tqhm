"""my_tutte.m -- Tutte embedding onto a disk of matching area.

MATLAB's `bfsearch(adjacency_matrix(O), O(1))` walks the boundary loop; on a
single cycle the BFS order alternates between the two directions
(0,1,last,2,...), so the loop is traced explicitly here instead -- that is what
the MATLAB code relies on (it feeds the result to cumsum of consecutive edge
lengths).

The `fill_holes` lines are commented out in the MATLAB source (FHF = F), so no
hole filling is needed here either.
"""

import numpy as np
import scipy.sparse as sp

from boundary_faces import boundary_faces
from edge_lengths import edge_lengths
from facet_adjacency_matrix import facet_adjacency_matrix
from conncomp import conncomp
from normrow import normrow
from doublearea import doublearea
from kharmonic import kharmonic


def _trace_loop(O):
    """Vertices in order around the single closed loop of edges O."""
    nxt = {}
    for a, b in O:
        nxt.setdefault(int(a), []).append(int(b))
        nxt.setdefault(int(b), []).append(int(a))

    start = int(O[0, 0])
    b = [start]
    prev = -1
    cur = start
    while True:
        a, c = nxt[cur]
        nx = a if a != prev else c
        if nx == start:
            break
        b.append(nx)
        prev, cur = cur, nx
    return np.array(b, dtype=np.int64)


def my_tutte(V, F):
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)

    # find boundaries
    O = boundary_faces(F)
    l = edge_lengths(V, O)
    # find longest boundary
    _, C = conncomp(facet_adjacency_matrix(O))
    Cl = np.bincount(C, weights=l)
    longest = int(np.argmax(Cl))

    FHF = F

    # Only keep longest boundary
    O = O[C == longest, :]
    # vertices in order around boundary
    b = _trace_loop(O)
    # parametric distance along boundary
    D = np.cumsum(np.r_[0.0, normrow(V[b, :] - V[b[np.r_[1:b.size, 0]], :])])
    D = D[:-1] / D[-1]
    # points on circle to map to
    bc = np.stack([np.cos(D * 2 * np.pi), np.sin(D * 2 * np.pi)], axis=1)

    # scale so that area matches
    total_area = np.sum(doublearea(V, F)) * 0.5
    bc = bc * np.sqrt(total_area / np.pi)

    # Cascading attempts to build the mapping to the disk: cotangent Laplacian,
    U = kharmonic(V, FHF, b, bc, 1)
    if doublearea(U, F).min() < 0:
        # intrinsic Laplacian; this "should" work but due to numerics might not
        U = kharmonic(V, FHF, b, bc, 1, IntrinsicDelaunay=True)
    if doublearea(U, F).min() < 0:
        U = kharmonic(None, FHF, b, bc, 1)

    assert doublearea(U, F).min() > 0
    return U
