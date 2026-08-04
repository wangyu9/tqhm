"""half_symmetric_rectangle_mesh.m -- grid mesh whose diagonals mirror about the
vertical mid-line: `\\` on the left half, `/` on the right.

(hm,hn) is half the number of cells horizontally and the full count vertically
(only the horizontal direction is mirrored, unlike symmetric_rectangle_mesh2).
"""

import numpy as np


def half_symmetric_rectangle_mesh(x0, y0, dx, dy, hm, hn):
    hm = int(hm)
    hn = int(hn)

    m = 2 * hm
    n = hn

    rows = []
    for j in range(1, n + 2):
        x = x0 + np.arange(m + 1) * dx
        y = (y0 + (j - 1) * dy) * np.ones(m + 1)
        rows.append(np.stack([x, y], axis=1))
    V = np.concatenate(rows, axis=0)

    # delaunay() does not give a symmetric mesh, so the strips are built by hand
    blocks = []
    for j in range(1, hn + 1):
        index_delta_x = 1
        index_delta_y = m + 1

        leftdown = (j - 1) * (m + 1)
        ld = leftdown + np.arange(hm)
        rd = ld + index_delta_x
        lu = ld + index_delta_y
        ru = rd + index_delta_y
        # \ style
        blocks.append(np.concatenate([np.stack([ld, rd, ru], axis=1),
                                      np.stack([ld, ru, lu], axis=1)], axis=0))

        leftdown = (j - 1) * (m + 1) + hm
        ld = leftdown + np.arange(hm)
        rd = ld + index_delta_x
        lu = ld + index_delta_y
        ru = rd + index_delta_y
        # / style
        blocks.append(np.concatenate([np.stack([ld, rd, lu], axis=1),
                                      np.stack([rd, ru, lu], axis=1)], axis=0))

    F = np.concatenate(blocks, axis=0).astype(np.int64)
    return V, F
