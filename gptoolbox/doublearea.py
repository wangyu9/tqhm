"""doublearea.m -- twice the area of each triangle (signed when dim==2)."""

import numpy as np
import igl


def doublearea(V, F):
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F)

    if V.shape[1] == 2:
        # igl.doublearea returns unsigned values, but gptoolbox's 2D branch is
        # signed, and the sign is what detects flipped triangles.
        r = V[F[:, 0], :] - V[F[:, 2], :]
        s = V[F[:, 1], :] - V[F[:, 2], :]
        return r[:, 0] * s[:, 1] - r[:, 1] * s[:, 0]

    if V.shape[1] == 3:
        return igl.doublearea(V, np.asarray(F, dtype=np.int32))

    l = np.stack([
        np.linalg.norm(V[F[:, 1], :] - V[F[:, 2], :], axis=1),
        np.linalg.norm(V[F[:, 2], :] - V[F[:, 0], :], axis=1),
        np.linalg.norm(V[F[:, 0], :] - V[F[:, 1], :], axis=1),
    ], axis=1)
    return igl.doublearea_intrinsic(l)
