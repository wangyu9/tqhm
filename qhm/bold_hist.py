"""bold_hist.m -- histogram with thick axes, sized for a paper figure.

MATLAB's `histogram` auto-binning rule is not reproduced exactly; matplotlib's
'auto' (max of Sturges and Freedman-Diaconis) is the closest built-in. The
PaperSize/PaperPositionMode bookkeeping becomes figure size + tight_layout, which
is what it achieves.
"""

import numpy as np

from tqhm_config import npy


def bold_hist(stats, xscale='linear'):
    import matplotlib.pyplot as plt

    p = {}

    h = plt.figure(figsize=(8, 6), dpi=100)

    ax = h.gca()

    hist = ax.hist(npy(stats).astype(np.float64).ravel(), bins='auto')

    if xscale:
        ax.set_xscale(xscale)

    ax.tick_params(labelsize=20, width=2)
    for s in ax.spines.values():
        s.set_linewidth(2)

    # box on
    for s in ax.spines.values():
        s.set_visible(True)

    if True:
        # tight
        h.tight_layout(pad=0)

    p['h'] = h
    p['hist'] = hist
    return p
