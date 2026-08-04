"""print_frames.m -- write selected history frames to PNG.

`frames` holds MATLAB 1-based indices into `history`, kept 1-based so the file
names match the MATLAB output. MATLAB's `savefig` (.fig) has no matplotlib
counterpart; the figure is pickled instead, which is the closest re-openable
format.
"""

import pickle
from pathlib import Path

import numpy as np

from tqhm_config import npy


def print_frames(history, F, frames, path):
    import matplotlib.pyplot as plt
    from render_mesh2 import render_mesh2

    for ii in range(len(np.atleast_1d(frames))):
        index = int(np.atleast_1d(frames)[ii])

        u = npy(history[index - 1]['u']).ravel()
        v = npy(history[index - 1]['v']).ravel()

        fig = plt.figure(figsize=(10, 10), dpi=100)
        t, l, h = render_mesh2(np.stack([u, v], axis=1), F,
                               EdgeColor=[0, 0, 0], FaceColor=[1, 1, 1])
        t.set(LineWidth=2)
        t.axes.set_axis_off()
        t.axes.set_aspect('equal')

        fname = str(Path(path) / str(index))

        with open(fname + '.fig.pkl', 'wb') as fh:
            pickle.dump(fig, fh)
        fig.savefig(fname + '.png')
        # im = imread(fname); im = imrotate(im,-90); imwrite(im,fname);
        # image_white2none([fname,'.png'],[fname,'.png']);
        plt.close(fig)
