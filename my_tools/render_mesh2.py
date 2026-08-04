"""render_mesh2.m -- flat-shaded 2D/3D mesh via tsurf.

The `.m` file's whole light/camera section is commented out, so `l` and `h` come
back empty, as they do in MATLAB.

`F=None` (from core_optimize_block when args.vis is set) is a hole in the MATLAB
script too: it passes `F` from the caller's workspace, which this port does not
thread through -- raise rather than draw the wrong thing.
"""

import numpy as np

from tqhm_config import npy


def render_mesh2(V, F, view=None, LightSource='default', FaceColor=None,
                 FaceScaleColor=None, ScaleColor=None, ColorMap='weights-neg',
                 ColorAxis=None, EdgeColor='none', LightMultiplier=1,
                 LineWidth=None):
    from tsurf import tsurf
    from my_colormap import my_colormap, listed

    # default values if not import from varargin
    azel = [0, 0] if view is None else view
    C = np.array([150, 220, 150]) / 255.0 * 1.1 if FaceColor is None else FaceColor
    w = ScaleColor
    c_range = ColorAxis
    uf = FaceScaleColor
    cmap = ColorMap

    FaceLighting = 'phong'

    if F is None:
        raise NotImplementedError('render_mesh2 needs F; the caller does not have it')

    V = npy(V)
    F = npy(F)

    if uf is not None and np.size(uf) != 0:
        uf = npy(uf).astype(np.float64).ravel()
        uf = (uf - uf.min()) / (uf.max() - uf.min())

        assert uf.min() >= 0
        assert uf.max() <= 1
        # C = value2color(u);
        ccc = my_colormap(cmap)

        C = ccc[np.minimum((uf * ccc.shape[0]).astype(np.int64), ccc.shape[0] - 1), :]

        # this is a temp hack, do it right later.
        # (MATLAB writes int8(255*C)+1, which saturates at 127 and flattens the
        # upper half of the ramp; uint8 here keeps the colors the hack intended.)
        C = np.clip(np.rint(255 * C), 0, 254).astype(np.uint8) + 1

    if w is None or np.size(w) == 0:
        t = tsurf(F, V, EdgeColor=EdgeColor, FaceColor=C,
                  FaceLighting=FaceLighting, SpecularStrength=1.0)
    else:
        w = npy(w).ravel()
        t = tsurf(F, V, EdgeColor=EdgeColor, FaceColor='interp',
                  FaceLighting=FaceLighting)
        t.set_colormap(listed(my_colormap(cmap)))
        if c_range is not None and np.size(c_range) != 0:
            t.set(CLim=(c_range[0], c_range[1]))
        if np.size(w) != 0:
            t.set(CData=w)
        _drawnow()

    if LineWidth is not None:
        t.set(LineWidth=LineWidth)

    l = []
    h = []

    return t, l, h


def _drawnow():
    import matplotlib.pyplot as plt
    if plt.isinteractive():
        plt.draw()
        plt.pause(0.001)
