"""draw_uv_with_distortion.m -- UV mesh with per-face distortion shading."""

import numpy as np

from tqhm_config import npy


def draw_uv_with_distortion(u, v, F, uf):
    from render_mesh2 import render_mesh2

    u = npy(u).ravel()
    v = npy(v).ravel()
    F = npy(F)
    uf = npy(uf).astype(np.float64).reshape(-1, 1)

    f = F.shape[0]

    if False:
        from doublearea import doublearea
        flipped = doublearea(np.stack([u, v], axis=1), F) < 0

        C = (np.array([1, 1, 1]) * np.ones((f, 3))) * (1 - flipped[:, None]) \
            + (np.array([0.8, 0.2, 0.2]) * np.ones((f, 3))) * flipped[:, None]
        t, l, h = render_mesh2(np.stack([u, v], axis=1), F,
                              EdgeColor=[0, 0, 0], FaceColor=[1, 1, 1])
        t.set(LineWidth=0.9)
        t.set(FaceVertexCData=C)
        t.axes.set_axis_off()
        t.axes.set_aspect('equal')

    t, l, h = render_mesh2(np.stack([u, v], axis=1), F, EdgeColor=[0, 0, 0])
    # ,'ColorMap','heat','FaceScaleColor',uf

    C = uf * np.array([1, 0, 0]) + np.array([1, 1, 1]) * (1 - uf)

    t.set(FaceColor='flat', FaceVertexCData=C, CDataMapping='scaled')

    # t.set(LineWidth=0.9)
    # t.set(FaceVertexCData=C)
    t.axes.set_axis_off()
    t.axes.set_aspect('equal')

    if False:
        t, l, h = render_mesh2(np.stack([u, v], axis=1), F[flipped, :],
                              EdgeColor=[0, 0, 0], FaceColor=[0.8, 0.2, 0.2],
                              ColorMap='heat', FaceScaleColor=uf)

        from render_mesh3 import render_mesh3
        t, l, _ = render_mesh3(np.stack([u, v], axis=1), F, EdgeColor=[0, 0, 0],
                               ColorMap='heat',
                               FaceColor=uf * np.array([1, 0, 0])
                               + np.array([1, 1, 1]) * (1 - uf))

        t, l, _ = render_mesh3(np.stack([u, v], axis=1), F, EdgeColor=[0, 0, 0],
                               ColorMap='heat', FaceScaleColor=uf)
