"""bold_hist2.m -- bold_hist with a fixed 0.001 bin width."""

import numpy as np

from tqhm_config import npy


def bold_hist2(stats, xscale='linear'):
    import matplotlib.pyplot as plt

    p = {}

    h = plt.figure(figsize=(8, 6), dpi=100)

    ax = h.gca()

    x = npy(stats).astype(np.float64).ravel()
    # 'BinWidth', 0.001
    bw = 0.001
    lo = np.floor(x.min() / bw) * bw
    hi = np.ceil(x.max() / bw) * bw
    hist = ax.hist(x, bins=np.arange(lo, hi + bw / 2, bw))

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
    return p
