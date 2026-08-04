"""my_colormap.m -- named colormaps, returned as an (N,3) float array.

MATLAB returns the raw colormap matrix and the callers hand it to `colormap()`;
here they wrap it with `listed()`.

`RdYlBu(800)` is a gptoolbox extra that is not part of this port; matplotlib's
'RdYlBu' is used instead, which is the same ColorBrewer ramp.
"""

import numpy as np


def _sampled(name, n):
    """MATLAB's `jet(n)` / `hot(n)`: the named ramp resampled to n rows."""
    import matplotlib as mpl
    cmap = mpl.colormaps[name]
    return np.asarray(cmap(np.linspace(0.0, 1.0, n)))[:, :3]


def listed(cmap):
    """(N,3) array -> a matplotlib colormap object."""
    from matplotlib.colors import ListedColormap
    return ListedColormap(np.asarray(cmap, dtype=np.float64))


def my_colormap(map_type):
    # remember to have: caxis([-0.2,1]);

    if map_type == 'weights-neg':
        lambda1 = (np.arange(1, 141) / 140.0)[:, None]
        lambda2 = 1 - lambda1
        red = np.array([[1.0, 0, 0]])
        green = np.array([[0, 1.0, 0]])
        blue = np.array([[0, 0, 1.0]])
        magenta = np.array([[1.0, 0, 1.0]])

        map_ = np.vstack([
            lambda1 * blue + lambda2 * magenta,
            lambda1 * green + lambda2 * blue,
            lambda1 * red + lambda2 * green,
        ])
        deepred_to_blue = _sampled('jet', 800)[100:800, :]
        exjet = np.vstack([
            lambda1 * blue + lambda2 * magenta,
            deepred_to_blue,
        ])
    elif map_type == 'RdYlBu':
        exjet = _sampled('RdYlBu', 800)
        exjet = exjet[::-1, :]
    elif map_type == 'heat':
        c = _sampled('hot', 256)
        exjet = c[::-1, :]
    elif map_type == 'halfHot':
        c = 0.77 * _sampled('hot', 256)
        exjet = c   # c[::-1, :]
    else:
        # MATLAB's colormap('default') is parula; matplotlib's is viridis.
        import matplotlib as mpl
        name = mpl.rcParams['image.cmap'] if map_type == 'default' else map_type
        exjet = _sampled(name, 256)

    return exjet
