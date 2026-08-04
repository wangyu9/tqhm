"""extrinsic_edge_hessians.m -- d^2(edge length squared)/dv^2 for each of the
three edges of a triangle, in the 6x6 layout (x1,y1,x2,y2,x3,y3).

Constant matrices, so they stay in numpy; callers move them to torch once.
"""

import numpy as np


def extrinsic_edge_hessians():
    core = 2 * np.array([[1, 0, -1, 0],
                         [0, 1, 0, -1],
                         [-1, 0, 1, 0],
                         [0, -1, 0, 1]], dtype=np.float64)

    He3 = np.zeros((6, 6))
    He3[0:4, 0:4] = core

    He1 = np.zeros((6, 6))
    He2 = np.zeros((6, 6))

    He1[2:6, 2:6] = core
    He2[np.ix_([0, 1, 4, 5], [0, 1, 4, 5])] = core

    return He1, He2, He3
