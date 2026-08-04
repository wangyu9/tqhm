"""play_history.m -- step through the optimization history, two seconds a frame.

A MATLAB script reading `history` and `F` from the caller's workspace; here a
function taking them as arguments.
"""

import numpy as np

from tqhm_config import npy


def play_history(history, F):
    import matplotlib.pyplot as plt
    from render_mesh2 import render_mesh2

    for ii in range(len(history)):
        u = npy(history[ii]['u']).ravel()
        v = npy(history[ii]['v']).ravel()

        t, l, h = render_mesh2(np.stack([u, v], axis=1), F,
                               EdgeColor=[0, 0, 0], FaceColor=[1, 1, 1])
        t.axes.set_axis_off()
        t.axes.set_aspect('equal')

        plt.pause(2)

        t.axes.legend([t.artist], ['frame: %d' % (ii + 1)])
