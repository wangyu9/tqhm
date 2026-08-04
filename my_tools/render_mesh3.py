"""render_mesh3.m -- shaded 3D mesh with a fixed view rotation.

matplotlib has no light objects, no phong/gouraud face lighting and no
specular/diffuse material properties, so `LightSource`, `Shading`,
`SpecularStrength` & co. are accepted and stored on the tsurf handle but do not
affect the image; `l` comes back empty instead of holding eight light handles.
Ambient occlusion still shades the vertex colors, via igl.

`h` is only assigned inside the `if false` block in the .m file, so it is None on
the normal path -- same as MATLAB, where requesting it would error.
"""

import numpy as np

from tqhm_config import npy


def _axisangle2matrix(w, a):
    """Rodrigues rotation; gptoolbox's axisangle2matrix is not part of this port."""
    w = np.asarray(w, dtype=np.float64)
    w = w / np.linalg.norm(w)
    K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * (K @ K)


def render_mesh3(V, F, view=None, AmbientOcclusion=None, Shading=1.0,
                 LightSource='default', FaceColor=None, FaceScaleColor=None,
                 VertexColor=None, ScaleColor=None, ColorMap='weights-neg',
                 ColorAxis=None, Quiver=None, EdgeColor='none',
                 LightMultiplier=1, LineWidth=None):
    import matplotlib.pyplot as plt
    from tsurf import tsurf
    from my_colormap import my_colormap, listed

    # default values if not import from varargin
    azel = view
    C = FaceColor          # do not change this.
    # C = [150,220,150]./255.*1.1;
    u = ScaleColor
    uf = FaceScaleColor
    c_range = ColorAxis

    AO = AmbientOcclusion
    cmap = ColorMap
    FaceLighting = 'gouraud'   # 'phong'

    VP = None
    VN = None
    if Quiver is not None:
        Quiver = npy(Quiver)
        assert Quiver.shape[1] == 6
        VP = Quiver[:, 0:3]
        VN = Quiver[:, 3:6]

    V = npy(V).astype(np.float64)
    F = npy(F).astype(np.int64)

    if V.shape[1] == 2:
        import warnings
        warnings.warn('z axis is missing! Append zero')
        V = np.c_[V, np.zeros(V.shape[0])]

    h = None
    if False:
        h = plt.figure(figsize=(8, 9))
        h.set_size_inches(16, 12)

    # t = tsurf(F,V,'EdgeColor',EdgeColor,'FaceColor','interp',...);
    R = _axisangle2matrix([0, 0, 1], np.pi) @ _axisangle2matrix([1, 0, 0], np.pi / 2)
    VD = V @ R

    empty_C = C is None or np.size(C) == 0
    empty_uf = uf is None or np.size(uf) == 0

    # MATLAB reads the *current* figure colormap here, before the colormap() call
    # at the end of the function; that is the default one unless a previous plot
    # changed it.
    current_cmap = my_colormap('default')

    if empty_C and empty_uf:
        t = tsurf(F, VD, EdgeColor=EdgeColor, FaceColor='interp',
                  FaceLighting=FaceLighting)

        if u is not None and np.size(u) != 0:
            u = npy(u).astype(np.float64).ravel()
            u = (u - u.min()) / (u.max() - u.min())
            assert u.min() >= 0
            assert u.max() <= 1
            if False:
                t.set(CData=u, CDataMapping='direct', CLim=(0, 1))
            else:
                # VertexColor = value2color(u);
                VertexColor = _ind2rgb(u, current_cmap)

        if VertexColor is not None and np.size(VertexColor) != 0:
            try:
                if AO is None or np.size(AO) == 0:
                    import igl
                    F32 = F.astype(np.int32)
                    AO = igl.ambient_occlusion(V, F32, V,
                                               igl.per_vertex_normals(V, F32), 1000)
            except Exception:
                AO = np.ones(V.shape[0])

            tmp = npy(VertexColor) * (1 - Shading * np.asarray(AO).reshape(-1, 1))
            t.set(FaceColor='interp', CData=tmp, CDataMapping='scaled')
    else:
        t = tsurf(F, VD, EdgeColor=EdgeColor, FaceColor='flat',
                  FaceLighting=FaceLighting)
        if empty_uf:
            C = npy(C)
            assert C.shape[0] == F.shape[0]
            FaceColor_ = C
        else:
            uf = npy(uf).astype(np.float64).ravel()
            uf = (uf - uf.min()) / (uf.max() - uf.min())
            assert uf.min() >= 0
            assert uf.max() <= 1
            FaceColor_ = _ind2rgb(uf, current_cmap)
        t.set(FaceColor='flat', FaceVertexCData=FaceColor_, CDataMapping='scaled')

    t.set_colormap(listed(my_colormap(cmap)))
    if False:
        if c_range is not None and np.size(c_range) != 0:
            t.set(CLim=(c_range[0], c_range[1]))

    t.SpecularStrength = .4          # 0.3
    t.DiffuseStrength = .45          # 0.1
    t.AmbientStrength = .6           # 0.7
    t.SpecularColorReflectance = .3
    t.SpecularExponent = 7

    l = []
    if LightSource == 'none':
        pass
    elif LightSource == 'default':
        # eight infinite lights; matplotlib has no equivalent
        s = 0.3 * LightMultiplier
        l = []

    ax = t.axes
    if ax.name == '3d':
        ax.set_proj_type('persp')
    ax.set_aspect('equal')
    ax.grid(False)
    if azel is not None and np.size(azel) != 0:
        _view(ax, azel)
    ax.set_axis_off()

    plt.gcf().set_facecolor('w')

    if VP is not None and np.size(VP) != 0:
        VP = VP @ R
        VN = VN @ R
        ax.quiver(VP[:, 0], VP[:, 1], VP[:, 2], VN[:, 0], VN[:, 1], VN[:, 2],
                  arrow_length_ratio=0.0)

    if V[:, 2].max() - V[:, 2].min() < 1e-10:
        _view(ax, [0, 0])

    if LineWidth is not None:
        t.set(LineWidth=LineWidth)

    if plt.isinteractive():
        plt.draw()
        plt.pause(0.001)

    return t, l, h


def _ind2rgb(u, cmap):
    """MATLAB squeeze(ind2rgb(floor(u*size(cmap,1))+1, cmap)) for u in [0,1]."""
    cmap = np.asarray(cmap)
    idx = np.minimum((np.asarray(u) * cmap.shape[0]).astype(np.int64), cmap.shape[0] - 1)
    return cmap[idx, :]


def _view(ax, azel):
    """MATLAB view([az,el]); its azimuth is measured 90 degrees off matplotlib's."""
    if ax.name != '3d':
        return
    ax.view_init(azim=float(azel[0]) - 90.0, elev=float(azel[1]))
