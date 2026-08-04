"""draw_uv_with_flips.m -- UV mesh with flipped triangles filled red."""

import numpy as np

from tqhm_config import npy
from doublearea import doublearea


def draw_uv_with_flips(u, v, F):
    from render_mesh2 import render_mesh2

    u = npy(u).ravel()
    v = npy(v).ravel()
    F = npy(F)
    UV = np.stack([u, v], axis=1)

    f = F.shape[0]

    flipped = doublearea(UV, F) < 0

    # flipped = V(F(:,1),1) > 0;
    C = (np.array([1, 1, 1]) * np.ones((f, 3))) * (1 - flipped[:, None]) \
        + (np.array([0.8, 0.2, 0.2]) * np.ones((f, 3))) * flipped[:, None]
    # C = C * 255;
    t, l, h = render_mesh2(UV, F, EdgeColor=[0, 0, 0], FaceColor=[1, 1, 1])
    t.set(LineWidth=2)
    t.set(FaceVertexCData=C)
    t.axes.set_axis_off()
    t.axes.set_aspect('equal')
    # hold on

    t, l, h = render_mesh2(UV, F[flipped, :], EdgeColor=[0, 0, 0],
                           FaceColor=[0.8, 0.2, 0.2])
