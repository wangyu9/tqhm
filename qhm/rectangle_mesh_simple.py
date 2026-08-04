"""rectangle_mesh_simple.m -- grid vertices triangulated by Delaunay.

(x0,y0) is the lower-left corner, (dx,dy) the cell size, (m,n) the number of
cells in each direction. The normal points along +z under the right hand rule.

MATLAB's `delaunay` becomes scipy.spatial.Delaunay, which is also CCW-oriented in
2D. On a regular grid every cell is a degenerate (cocircular) quad, so which
diagonal each library picks is arbitrary and the two triangulations will differ
in general -- use symmetric_rectangle_mesh2 / half_symmetric_rectangle_mesh when
the diagonal pattern matters.
"""

import numpy as np
from scipy.spatial import Delaunay


def rectangle_mesh_simple(x0, y0, dx, dy, m, n):
    m = int(m)
    n = int(n)

    rows = []
    for j in range(1, n + 2):
        x = x0 + np.arange(m + 1) * dx
        y = (y0 + (j - 1) * dy) * np.ones(m + 1)
        rows.append(np.stack([x, y], axis=1))
    V = np.concatenate(rows, axis=0)

    F = np.asarray(Delaunay(V).simplices, dtype=np.int64)
    return V, F
