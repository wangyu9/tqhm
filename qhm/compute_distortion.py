"""compute_distortion.m -- per-face distortion from the intrinsic gradient.

The MATLAB source is an unfinished stub: it stops mid-statement at `wf(j) = `
(so the .m file does not even parse), and the line above it indexes column 2 of
the column vector `Gv`. Everything up to that point is transcribed; the body of
the loop raises NotImplementedError rather than inventing a distortion measure.

`jacobians(G,u,v)` exposes the part that *is* well defined: the per-face 2x2
Jacobian [[Gxu,Gyu],[Gxv,Gyv]] of the map, which is what the stub was assembling.
"""

import numpy as np
import torch


def jacobians(G, u, v):
    """Per-face 2x2 Jacobians of the map (u,v); shape (f,2,2)."""
    f = G.shape[0] // 2
    n = u.shape[0] if hasattr(u, 'shape') else len(u)
    assert G.shape[1] == n

    Gu = G @ u
    Gv = G @ v

    stack = torch.stack if torch.is_tensor(Gu) else np.stack
    # rows: the gradient of u then of v (MATLAB's `note the transpose`)
    return stack([stack([Gu[:f], Gu[f:]], axis=-1),
                  stack([Gv[:f], Gv[f:]], axis=-1)], axis=-2)


def compute_distortion(G, u, v):
    f = G.shape[0] // 2

    n = u.shape[0] if hasattr(u, 'shape') else len(u)

    assert G.shape[1] == n

    Gu = G @ u
    Gv = G @ v

    wf = np.zeros(f)

    for j in range(f):
        Jj = np.stack([Gu[[j, j + f]], Gv[[j, j + f]]], axis=1).T   # note the transpose
        _ = Jj

        raise NotImplementedError(
            'compute_distortion.m is an unfinished stub: `wf(j) = ` has no '
            'right-hand side')

    return wf
