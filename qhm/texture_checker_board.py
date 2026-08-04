"""texture_checker_board.m -- s-pixel checkerboard texture, uint8 RGB.

`meshgrid(1:m,1:n)` is (n,m) while `im` is (m,n,3), so the .m file only runs for
m == n; the shapes are kept as written and numpy raises the same way MATLAB does.
"""

import numpy as np


def texture_checker_board(m, n, s, color):
    x = np.arange(1, m + 1)
    y = np.arange(1, n + 1)
    X, Y = np.meshgrid(x, y)

    # s = 16;

    F = np.mod(np.ceil(X / s) + np.ceil(Y / s), 2)
    # 0.3, 0.3, 0.8

    im = np.ones((m, n, 3))
    im[:, :, 0] = color[0] * 255 * F
    im[:, :, 1] = color[1] * 255 * F
    im[:, :, 2] = color[2] * 255 * F

    return np.clip(im, 0, 255).astype(np.uint8)
