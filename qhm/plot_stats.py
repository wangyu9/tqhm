"""plot_stats.m -- histogram of the singular-value ratio on a log x axis."""

import numpy as np

from tqhm_config import npy


def plot_stats(stats, metric=None, xscale='log'):
    import matplotlib.pyplot as plt

    p = {}

    h = plt.figure(figsize=(8, 6), dpi=100)

    ax = h.gca()

    hist = ax.hist(npy(stats['sigma_ratio']).astype(np.float64).ravel(), bins='auto')

    if xscale:
        ax.set_xscale(xscale)

    ax.tick_params(labelsize=20, width=2)
    for s in ax.spines.values():
        s.set_linewidth(2)
        s.set_visible(True)

    if True:
        # tight
        h.tight_layout(pad=0)

    p['h'] = h
    p['hist'] = hist

    if False:
        # line plot of the per-iteration values in `u`, with custom line colors
        p = {}

        h = plt.figure(figsize=(14.4, 10.8), dpi=100)

        line_colors = None
        if line_colors is not None and np.size(line_colors) != 0:
            plt.rcParams['axes.prop_cycle'] = plt.cycler(color=line_colors)

        ax = h.gca()
        ax.tick_params(labelsize=20, width=2)
        for s in ax.spines.values():
            s.set_linewidth(2)

        u = np.atleast_2d(npy(u))
        if u.shape[0] == 1:
            u = u.T

        c = u.shape[1]
        ls = [None] * c
        for i in range(c):
            ls[i] = ax.plot(np.arange(u.shape[0]), u[:, i], linewidth=3)

        # ax.set_ylim(0, 35)

        if False:
            # tight
            h.tight_layout(pad=0)

        p['ls'] = ls
        p['h'] = h

    return p
