"""tsurf.m -- trisurf wrapper.

The one gptoolbox file that is *not* delegated to igl: igl has no plotting layer,
so this is a matplotlib reimplementation. 2D meshes become a
`matplotlib.tri.Triangulation` (PolyCollection for flat shading, TriMesh for
'interp'), 3D meshes a `Poly3DCollection`.

MATLAB's name/value pairs arrive as keyword arguments and the returned handle
mimics the bits of a MATLAB surface handle the callers touch:
`t.set(LineWidth=2)`, `t.set(FaceColor='flat', FaceVertexCData=C)`,
`t.Vertices`, `t.Faces`. Lighting properties (FaceLighting, SpecularStrength,
DiffuseStrength, ...) have no matplotlib equivalent; they are stored on the
handle and ignored when drawing.
"""

import numpy as np

from tqhm_config import npy


def _get_axes(dim):
    import matplotlib.pyplot as plt
    fig = plt.gcf()
    ax = fig.gca()
    if dim == 3 and ax.name != '3d':
        # a fresh 2D axes can be swapped for a 3D one; a used one cannot
        if not ax.has_data():
            ax.remove()
            ax = fig.add_subplot(projection='3d')
    return ax


def _as_rgb(c):
    """Coerce a MATLAB color spec to something matplotlib accepts."""
    if isinstance(c, str):
        return c
    c = npy(c)
    if c.ndim == 2 and c.shape[0] == 3 and c.shape[1] != 3:
        c = c.T                      # render_mesh2 passes C'
    if np.issubdtype(c.dtype, np.integer) and c.size and np.abs(c).max() > 1:
        # render_mesh2's `int8(255*C)+1` hack; bring it back into [0,1]
        c = np.clip(c.astype(np.float64) / 255.0, 0.0, 1.0)
    c = np.clip(np.asarray(c, dtype=np.float64), 0.0, 1.0)
    if c.ndim == 1:
        return c
    if c.shape[0] == 1:
        return c[0]
    return c


class TsurfHandle:
    """The subset of a MATLAB surface handle that the callers here use."""

    def __init__(self, ax, V, F, dim):
        self.axes = ax
        self.Vertices = V
        self.Faces = F
        self.dim = dim
        self.artist = None
        self._face_color = None
        self._edge_color = 'none'
        self._line_width = 0.5
        self._cdata = None
        self._cmap = None
        self._clim = None
        self.props = {}

    # --- building ------------------------------------------------------

    def _tri_verts(self):
        V, F = self.Vertices, self.Faces
        return V[F, :] if self.dim == 3 else V[F, 0:2]

    def _build(self):
        ax = self.axes
        if self.artist is not None:
            self.artist.remove()
            self.artist = None

        fc, cdata = self._face_color, self._cdata
        gouraud = (isinstance(fc, str) and fc == 'interp' and self.dim == 2
                   and (cdata is None
                        or (np.ndim(cdata) == 1 and cdata.size == self.Vertices.shape[0])))

        if gouraud:
            from matplotlib.tri import Triangulation
            tri = Triangulation(self.Vertices[:, 0], self.Vertices[:, 1], self.Faces)
            z = np.zeros(self.Vertices.shape[0]) if cdata is None else np.asarray(cdata)
            art = ax.tripcolor(tri, z, shading='gouraud')
        else:
            verts = self._tri_verts()
            if self.dim == 3:
                from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                art = Poly3DCollection(verts)
            else:
                from matplotlib.collections import PolyCollection
                art = PolyCollection(verts)
            ax.add_collection(art)

            if cdata is not None:
                arr = np.asarray(cdata)
                if arr.ndim == 2:
                    rgb = _as_rgb(arr)
                    if np.ndim(rgb) == 2 and rgb.shape[0] == self.Vertices.shape[0] \
                            and rgb.shape[0] != self.Faces.shape[0]:
                        # per-vertex truecolor on a flat-shaded collection
                        rgb = rgb[self.Faces, :].mean(axis=1)
                    art.set_facecolor(rgb)
                else:
                    if arr.size == self.Vertices.shape[0]:
                        # no per-vertex shading on a (Poly3D)Collection: average
                        arr = arr[self.Faces].mean(axis=1)
                    art.set_array(arr)
                    art.autoscale()
            elif isinstance(fc, str):
                art.set_facecolor('none' if fc == 'none' else (0.8, 0.8, 0.8))
            else:
                art.set_facecolor(_as_rgb(fc))

        art.set_edgecolor(_as_rgb(self._edge_color))
        art.set_linewidth(self._line_width)
        if self._cmap is not None:
            art.set_cmap(self._cmap)
        if self._clim is not None:
            art.set_clim(*self._clim)
        self.artist = art

        self.axis_tight()
        return art

    def axis_tight(self):
        """MATLAB `axis tight`. add_collection does not autoscale a 3D axes."""
        ax, V = self.axes, self.Vertices
        if ax.name == '3d':
            lo, hi = V.min(axis=0), V.max(axis=0)
            span = np.where(hi - lo > 0, hi - lo, 1.0)
            ax.set_xlim(lo[0], lo[0] + span[0])
            ax.set_ylim(lo[1], lo[1] + span[1])
            ax.set_zlim(lo[2], lo[2] + span[2])
        else:
            ax.autoscale_view()

    # --- MATLAB-style property setting ---------------------------------

    def set(self, **kw):
        rebuild = False
        for name, value in kw.items():
            if name == 'LineWidth':
                self._line_width = value
                if self.artist is not None:
                    self.artist.set_linewidth(value)
            elif name == 'EdgeColor':
                self._edge_color = value
                if self.artist is not None:
                    self.artist.set_edgecolor(_as_rgb(value))
            elif name == 'FaceColor':
                self._face_color = value
                rebuild = True
            elif name in ('CData', 'FaceVertexCData'):
                self._cdata = None if value is None else npy(value)
                rebuild = True
            elif name == 'ColorMap':
                self.set_colormap(value)
            elif name in ('CAxis', 'ColorAxis', 'CLim'):
                self._clim = (float(value[0]), float(value[1]))
                if self.artist is not None:
                    self.artist.set_clim(*self._clim)
            elif name == 'CDataMapping':
                # 'scaled' is matplotlib's only mode; 'direct' would need the
                # colormap indexed by hand, which no caller here relies on.
                self.props[name] = value
            else:
                self.props[name] = value
        if rebuild and not self._can_update():
            self._build()

    def _can_update(self):
        """Cheap in-place update when only the scalar CData changed."""
        art, cdata = self.artist, self._cdata
        if art is None or cdata is None or np.ndim(cdata) != 1:
            return False
        want = art.get_array()
        if want is None or np.size(want) != np.size(cdata):
            return False
        art.set_array(np.asarray(cdata))
        art.autoscale()
        return True

    def set_colormap(self, cmap):
        self._cmap = cmap
        if self.artist is not None:
            self.artist.set_cmap(cmap)

    def __setattr__(self, name, value):
        # render_mesh3 assigns lighting properties directly: t.SpecularStrength = .4
        if name[:1].isupper() and name not in ('Vertices', 'Faces') \
                and 'props' in self.__dict__:
            self.props[name] = value
            return
        object.__setattr__(self, name, value)


def tsurf(F, V, VertexIndices=0, FaceIndices=0, Tets=None,
          ButtonDownFcn='default', **kw):
    V = npy(V).astype(np.float64)
    F = None if F is None else npy(F).astype(np.int64)

    vertex_indices = VertexIndices
    face_indices = FaceIndices
    tets = Tets

    # number of vertices
    n = V.shape[0]

    # number of dimensions
    dim = V.shape[1]

    if dim == 2 or (dim == 3 and np.abs(V[:, 2]).sum() == 0):
        V = np.stack([V[:, 0], V[:, 1], 0 * V[:, 0]], axis=1)
        dim = 2
    elif dim > 3 or dim < 2:
        raise ValueError('V must be #V x 3 or #V x 2')

    if tets is None or (np.ndim(tets) > 0 and np.size(tets) == 0):
        tets = False
        if F.shape[1] == 4:
            import igl
            VV = (V - V.min(axis=0)) * (V.max() - V.min())
            tets = igl.volume(VV, F.astype(np.int32)).sum() > 1e-10
            if not tets:
                Ftri = np.vstack([F[:, [0, 1, 2]], F[:, [0, 2, 3]]])
                Itri = np.tile(np.arange(F.shape[0]), 2)
        else:
            tets = False
            Ftri = F
            Itri = np.arange(F.shape[0])

    from barycenter import barycenter

    if tets:
        raise NotImplementedError('tetramesh has no matplotlib counterpart')
    else:
        ax = _get_axes(dim)
        t_copy = TsurfHandle(ax, V, Ftri, dim)
        t_copy._face_color = kw.pop('FaceColor', (0.8, 0.8, 0.8))
        t_copy._edge_color = kw.pop('EdgeColor', 'k')
        if 'CData' in kw:
            t_copy._cdata = npy(kw.pop('CData'))
        if 'FaceVertexCData' in kw:
            t_copy._cdata = npy(kw.pop('FaceVertexCData'))
        if 'LineWidth' in kw:
            t_copy._line_width = kw.pop('LineWidth')
        if 'ColorMap' in kw:
            t_copy._cmap = kw.pop('ColorMap')
        t_copy._build()

        FC = barycenter(V, Ftri) if face_indices else None
        if face_indices == 1:
            for j in range(Ftri.shape[0]):
                _text(ax, FC[j], str(j + 1), dim, bg=(.7, .7, .7))
        elif face_indices:
            for j in range(Ftri.shape[0]):
                _text(ax, FC[j], str(j + 1), dim)

    # if 2d then set to view (x,y) plane
    if dim == 2:
        ax.set_aspect('equal')

    if vertex_indices:
        visible = np.unique(Ftri).reshape(-1)
        bg = (.8, .8, .8) if vertex_indices == 1 else None
        for j in visible:
            _text(ax, V[j], str(j + 1), dim, bg=bg)

    # axis tight
    t_copy.axis_tight()

    if kw:
        t_copy.set(**kw)

    # the buttondown callback launches gptoolbox's meshplot, which is not ported
    if ButtonDownFcn == 'default':
        pass
    elif ButtonDownFcn == 'none':
        pass

    return t_copy


def _text(ax, p, s, dim, bg=None):
    kw = {} if bg is None else {'backgroundcolor': bg}
    if dim == 3:
        ax.text(p[0], p[1], p[2], s, **kw)
    else:
        ax.text(p[0], p[1], s, **kw)
