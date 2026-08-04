"""simple_deform_complex.m -- interactive 2D deformation widget: drag the control
points in C, the mesh follows through the weights W.

MATLAB's varargin name/value pairs become keyword arguments and the `g_Deform`
global becomes the returned object, whose `.R`, `.new_C` and `.update_positions()`
are the fields MATLAB documents as the global output. `gid` is still returned
first so callers that only wanted the index keep working.

The callbacks are matplotlib event handlers (`button_press_event`,
`motion_notify_event`, `button_release_event`, `key_press_event`) in place of
MATLAB's ButtonDownFcn / windowbuttonmotionfcn / windowbuttonupfcn / KeyPressFcn.
`get/set(handle,'Vertices')` has no matplotlib equivalent, so `new_V` is kept as
the state of record and the Poly3DCollection is rebuilt from it.

The 'LBS' and 'DQLBS' modes need gptoolbox's skinning helpers
(skinning_transformations, lbs, axisangle2quat, quattrans2udq, dualquatlbs,
stacktimes, pseudoedge_dof), none of which exist in this repo; those branches
raise NotImplementedError. 'LI', 'CLI' and 'visualizeSupport' -- the modes this
paper's code actually uses -- are live.
"""

import numpy as np


class _Deform:
    """Stands in for one entry of MATLAB's `g_Deform` struct array."""

    def __init__(self):
        self.R = None
        self.new_C = None
        self.update_positions = None
        self.tsh = None
        self.wvsh = None


def _rotate_around_center(input_points, angle):
    """added by wangyu"""
    input_points = np.asarray(input_points, dtype=np.float64)
    rotate_center = input_points.mean(axis=0)
    dx = input_points[:, 0] - rotate_center[0]
    dy = input_points[:, 1] - rotate_center[1]
    return np.stack([
        dx * np.cos(angle) - dy * np.sin(angle) + rotate_center[0],
        dx * np.sin(angle) + dy * np.cos(angle) + rotate_center[1],
    ], axis=1)


def simple_deform_complex(V, F, C, W, ShowWeightVisualization=False,
                          InterpMode='LBS', BoneEdges=None, CageEdges=None,
                          PointHandles=None, AutoDOF='none', PseudoEdges=None,
                          InitHandlePos=None):
    import matplotlib.pyplot as plt

    # --- parse input / set default parameters ---
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    C = np.asarray(C, dtype=np.float64)
    W = np.asarray(W)

    C0 = InitHandlePos
    # set default point handles
    P = np.arange(C.shape[0]) if PointHandles is None \
        else np.asarray(PointHandles, dtype=np.int64).ravel()
    # set default bone edge handles
    BE = np.zeros((0, 2), dtype=np.int64) if BoneEdges is None \
        else np.asarray(BoneEdges, dtype=np.int64)
    # be sure that control vertices are in 2D
    if C.shape[1] == 3:
        C = C[:, 0:2]
    # weights used for contours
    CW = W
    # weights used for weight visualization
    WVW = W
    show_weight_visualization = bool(ShowWeightVisualization)

    if InterpMode not in ('LBS', 'DQLBS', 'LI', 'CLI', 'visualizeSupport'):
        raise ValueError('InterpMode must be either LBS, DQLBS or LI')
    interp_mode = InterpMode

    auto_dof = AutoDOF
    PE = np.zeros((0, 2), dtype=np.int64) if PseudoEdges is None \
        else np.asarray(PseudoEdges, dtype=np.int64)
    CE = np.zeros((0, 2), dtype=np.int64) if CageEdges is None \
        else np.asarray(CageEdges, dtype=np.int64)

    # number of point handles
    np_ = P.size
    # number of bone handles
    nb = BE.shape[0]

    # --- initialize ---
    state = {'new_V': V.copy(), 'iP': np.array([], dtype=np.int64)}

    # --- prepare output ---
    d = _Deform()
    gid = 0
    # rotations stored at each control point; for 2D an np_ by 1 list of angles
    d.R = np.zeros(np_)

    # --- set up plots ---
    fig = plt.figure()
    fig.clf()
    number_of_subplots = 1
    current_subplot = 1
    if show_weight_visualization:
        number_of_subplots += 1

    ax = fig.add_subplot(1, number_of_subplots, 1)

    # plot the original mesh
    view_2D = True
    if view_2D:
        d.tsh = ax.tripcolor(V[:, 0], V[:, 1], F, np.zeros(V.shape[0]),
                             shading='gouraud')
        ax.set_aspect('equal')
    else:
        ax3 = fig.add_subplot(1, number_of_subplots, 1, projection='3d')
        d.tsh = ax3.plot_trisurf(V[:, 0], V[:, 1], V[:, 2], triangles=F)
        ax = ax3

    # plot bones
    B_plot_outer = B_plot_inner = None
    if nb > 0:
        # thick lines for bones (the outline of the lines)
        B_plot_outer = ax.plot(np.stack([C[BE[:, 0], 0], C[BE[:, 1], 0]]),
                              np.stack([C[BE[:, 0], 1], C[BE[:, 1], 1]]),
                              '-k', linewidth=5)
        # thin lines for bones (the inner line of the lines)
        B_plot_inner = ax.plot(np.stack([C[BE[:, 0], 0], C[BE[:, 1], 0]]),
                              np.stack([C[BE[:, 0], 1], C[BE[:, 1], 1]]),
                              '-b', linewidth=2)

    PE_plot = None
    if PE.shape[0] > 0:
        PE_plot = ax.plot(np.stack([C[P[PE[:, 0]], 0], C[P[PE[:, 1]], 0]]),
                          np.stack([C[P[PE[:, 0]], 1], C[P[PE[:, 1]], 1]]),
                          '--r', linewidth=5)

    CE_plot_outer = CE_plot_inner = None
    if CE.shape[0] > 0:
        CE_plot_outer = ax.plot(np.stack([C[P[CE[:, 0]], 0], C[P[CE[:, 1]], 0]]),
                                np.stack([C[P[CE[:, 0]], 1], C[P[CE[:, 1]], 1]]),
                                '-k', linewidth=5)
        CE_plot_inner = ax.plot(np.stack([C[P[CE[:, 0]], 0], C[P[CE[:, 1]], 0]]),
                                np.stack([C[P[CE[:, 0]], 1], C[P[CE[:, 1]], 1]]),
                                '-', color=[1, 1, 0.2], linewidth=2)

    # plot the control points
    C_plot = ax.scatter(C[:, 0], C[:, 1], s=100, marker='o',
                        facecolor=[0.9, 0.8, 0.1], edgecolor='k', linewidth=2,
                        zorder=3, picker=True)

    # set up the weight visualization plot
    if show_weight_visualization:
        current_subplot += 1
        assert CW.shape == W.shape
        axw = fig.add_subplot(1, number_of_subplots, current_subplot,
                              projection='3d')
        d.wvsh = axw.plot_trisurf(V[:, 0], V[:, 1], WVW[:, 0], triangles=F)
        axw.set_zlim(-1.3, 1.3)
        fig.colorbar(d.wvsh, ax=axw)
    else:
        axw = None

    if False:
        # the second visualization subplot, commented out in the MATLAB source
        current_subplot += 1
        assert CW.shape == W.shape
        axw2 = fig.add_subplot(1, number_of_subplots, current_subplot)
        d.wvsh = axw2.tripcolor(V[:, 0], V[:, 1], F, np.zeros(V.shape[0]),
                                shading='gouraud')
        axw2.set_aspect('equal')

    # --- set up interaction variables ---
    # window xmin, xmax, ymin, ymax
    win_min = np.minimum(C[:, 0:2].min(axis=0), V[:, 0:2].min(axis=0))
    win_max = np.maximum(C[:, 0:2].max(axis=0), V[:, 0:2].max(axis=0))
    itr = {'down_pos': None, 'drag_pos': None, 'last_drag_pos': None,
           'down_V': None, 'ci': None, 'down_type': '',
           'win_min': win_min, 'win_max': win_max, 'cids': []}

    if show_weight_visualization:
        print('\nCLICK a control point to visualize its corresponding '
              'weights on the mesh.')
    print('DRAG a control point to deform the mesh.\n'
          'RIGHT CLICK DRAG a control point to rotate point handles.\n')

    # --- callbacks for keyboard and mouse ---

    def update_positions():
        # update the display positions
        C_plot.set_offsets(d.new_C[:, 0:2])

        if nb > 0:
            for k, ln in enumerate(B_plot_outer):
                ln.set_data([d.new_C[BE[k, 0], 0], d.new_C[BE[k, 1], 0]],
                            [d.new_C[BE[k, 0], 1], d.new_C[BE[k, 1], 1]])
            for k, ln in enumerate(B_plot_inner):
                ln.set_data([d.new_C[BE[k, 0], 0], d.new_C[BE[k, 1], 0]],
                            [d.new_C[BE[k, 0], 1], d.new_C[BE[k, 1], 1]])
        # update the pseudo edge plots
        if PE.shape[0] > 0:
            for k, ln in enumerate(PE_plot):
                ln.set_data([d.new_C[P[PE[k, 0]], 0], d.new_C[P[PE[k, 1]], 0]],
                            [d.new_C[P[PE[k, 0]], 1], d.new_C[P[PE[k, 1]], 1]])
        # update the cage edge plots
        if CE.shape[0] > 0:
            for plots in (CE_plot_outer, CE_plot_inner):
                for k, ln in enumerate(plots):
                    ln.set_data([d.new_C[P[CE[k, 0]], 0], d.new_C[P[CE[k, 1]], 0]],
                                [d.new_C[P[CE[k, 0]], 1], d.new_C[P[CE[k, 1]], 1]])

        # update the mesh positions
        if interp_mode == 'LBS':
            # USING LINEAR BLEND SKINNING
            raise NotImplementedError(
                "InterpMode 'LBS' needs gptoolbox's skinning_transformations/lbs, "
                'which are not part of this repo')
        elif interp_mode == 'LI':
            # MATLAB computes TR = skinning_transformations(...) here (and
            # overrides it via pseudoedge_dof when auto_dof=='pseudoedges'), but
            # then never uses TR on this branch -- lbs is commented out. Only the
            # auto_dof path would need the missing gptoolbox helpers.
            if auto_dof == 'pseudoedges':
                raise NotImplementedError(
                    "AutoDOF 'pseudoedges' needs gptoolbox's pseudoedge_dof/"
                    'axisangle2quat, which are not part of this repo')
            state['new_V'] = W @ d.new_C[P, :]   # wangyu
        elif interp_mode == 'CLI':
            if auto_dof == 'pseudoedges':
                raise NotImplementedError(
                    "AutoDOF 'pseudoedges' needs gptoolbox's pseudoedge_dof/"
                    'axisangle2quat, which are not part of this repo')
            CP = d.new_C[P, :]                   # wangyu
            nv = W @ (CP[:, 0] + 1j * CP[:, 1])
            state['new_V'] = np.stack([nv.real, nv.imag], axis=1)
        elif interp_mode == 'DQLBS':
            # USING DUAL QUATERNION SKINNING
            raise NotImplementedError(
                "InterpMode 'DQLBS' needs gptoolbox's axisangle2quat/quattrans2udq/"
                'dualquatlbs/stacktimes, which are not part of this repo')
        elif interp_mode == 'visualizeSupport':
            pass   # do nothing

        # update the mesh positions
        new_V = state['new_V']
        d.tsh.remove()
        d.tsh = ax.tripcolor(new_V[:, 0], new_V[:, 1], F, np.zeros(new_V.shape[0]),
                             shading='gouraud')
        fig.canvas.draw_idle()

    d.update_positions = update_positions

    def oncontrolsup(event=None):
        # tell the window to handle drag and up events itself
        for cid in itr['cids']:
            fig.canvas.mpl_disconnect(cid)
        itr['cids'] = []

        cur_V = state['new_V'][:, 0:2]
        # scale the window to fit
        itr['win_min'] = np.minimum(itr['win_min'], cur_V.min(axis=0))
        itr['win_max'] = np.maximum(itr['win_max'], cur_V.max(axis=0))
        ax.set_xlim(itr['win_min'][0], itr['win_max'][0])
        ax.set_ylim(itr['win_min'][1], itr['win_max'][1])
        fig.canvas.draw_idle()

    def oncontrolsdrag(event):
        if event.xdata is None:
            return
        # keep the last drag position
        itr['last_drag_pos'] = itr['drag_pos']
        itr['drag_pos'] = np.array([event.xdata, event.ydata])
        delta = itr['drag_pos'] - itr['last_drag_pos']

        if itr['down_type'] == 'left':
            # move the selected control point group by the drag offset
            d.new_C[itr['iP'], :] = d.new_C[itr['iP'], :] + delta
        else:
            found = np.isin(itr['ci'], P)
            if found:
                iP = np.flatnonzero(P == itr['ci'])
                itr['iP'] = iP
                if interp_mode == 'LI':
                    d.new_C[iP, :] = _rotate_around_center(
                        d.new_C[iP, :], 2 * np.pi * delta[0] / 100)
                else:
                    d.R[iP] = d.R[iP] + 2 * np.pi * delta[0] / 100
        update_positions()
        # writeOBJ('temp.obj', new_V, F); writeDMAT('H_bc.dmat', new_C)

    def oncontrolsdown(event):
        if event.xdata is None or event.inaxes is not ax:
            return
        # get the current mouse position, and remember the old one
        itr['down_pos'] = np.array([event.xdata, event.ydata])
        itr['last_drag_pos'] = itr['down_pos']
        itr['drag_pos'] = itr['down_pos']
        # keep track of the control point positions at mouse down
        # (matplotlib keeps them in the collection's offsets)
        d.new_C = np.asarray(C_plot.get_offsets(), dtype=np.float64)
        # get the index of the closest control point
        ci = int(np.argmin(np.sum((d.new_C[:, 0:2] - itr['down_pos']) ** 2, axis=1)))
        itr['ci'] = ci
        # keep track of the mesh vertices at mouse down
        itr['down_V'] = state['new_V'][:, 0:2].copy()

        # drag and up events should be handled by the controls
        itr['cids'] = [
            fig.canvas.mpl_connect('motion_notify_event', oncontrolsdrag),
            fig.canvas.mpl_connect('button_release_event', oncontrolsup),
            fig.canvas.mpl_connect('key_press_event', onkeypress),
        ]
        itr['down_type'] = 'left' if event.button == 1 else 'right'

        # try to find ci in the list of point handles
        found = bool(np.isin(ci, P))
        temp_iP = np.flatnonzero(P == ci)
        found_in_iP = bool(np.isin(ci, itr['iP']))
        if not found_in_iP:
            itr['iP'] = temp_iP   # [iP; temp_iP]

        if found:
            # set the color of the mesh plot to the weights of the selected handle
            # d.tsh.set_array(W[:, iP])
            # change the weights in the weight visualization
            if show_weight_visualization:
                temp = np.sum(WVW[:, itr['iP']], axis=1)
                # set the weights that are exactly 0 to a different color
                index = np.flatnonzero(temp == 0)
                temp[index] = -100

                d.wvsh.remove()
                d.wvsh = axw.plot_trisurf(V[:, 0], V[:, 1], temp, triangles=F)
                fig.canvas.draw_idle()

    def onkeypress(event):
        ch = event.key
        if ch == 'r':
            d.new_C = C.copy()
            d.R = np.zeros(np_)
            update_positions()
        elif ch == 'u':
            update_positions()
        elif ch == 't':
            # d.new_C = C
            d.R = d.R + 0.1 * np.ones(np_)
            d.new_C = _rotate_around_center(d.new_C, 0.1)
            update_positions()
        elif ch == 'c':
            itr['iP'] = np.array([], dtype=np.int64)
        elif ch == 'a':
            if interp_mode == 'LI':
                d.new_C[itr['iP'], :] = _rotate_around_center(
                    d.new_C[itr['iP'], :], 2 * np.pi * 1 / 100)
            update_positions()
        oncontrolsup(event)

    fig.canvas.mpl_connect('button_press_event', oncontrolsdown)
    fig.canvas.mpl_connect('key_press_event', onkeypress)

    # added by wangyu: set the initial position of the handles and apply skinning
    d.new_C = None if C0 is None else np.asarray(C0, dtype=np.float64)
    if d.new_C is not None and d.new_C.shape[0] == C.shape[0]:
        update_positions()

    return gid, d
