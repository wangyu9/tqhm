"""distort_to_color.m -- map the singular-value ratio to a [0,1] color weight."""

import numpy as np

from tqhm_config import npy


def distort_to_color(stats, encoder, args=None):
    ratio = npy(stats['sigma_ratio']).astype(np.float64).copy()

    assert np.all(ratio >= 1)

    # encoder = 'log10';

    if encoder == 'temp':
        base = 50
        ratio[ratio > base] = base

        ratio = np.log(ratio) / np.log(base)

    elif encoder == 'log10':
        # log10 scale
        ratio[ratio > 10] = 10

        ratio = np.log10(ratio)

    elif encoder == 'linear10':
        ratio = (ratio - 1) / (10 - 1)

    wf = ratio
    return wf
