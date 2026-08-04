"""deformed_anisotropicity.m -- cotangent edge weights of the *deformed* mesh.

Reads the edge list out of pre.DEC, which triangle_mesh.m only fills when the
external DEC toolbox (getMeshData/discreteExteriorCalculus) is on the path. That
toolbox is not part of this repo, so the code path is kept and raises.

MATLAB reads LD entry by entry in a loop; here the whole edge list is gathered at
once, which is the same numbers.
"""

import numpy as np
import scipy.sparse as sp

from grad import grad
from doublearea import doublearea


def deformed_anisotropicity(pre, VD):
    F = pre['F']

    if 'DEC' not in pre:
        raise NotImplementedError(
            'deformed_anisotropicity.m needs pre.DEC.mesh.Elist, which comes from '
            'the external DEC toolbox that is not part of this repo')

    EL = pre['DEC']['mesh']['Elist']

    VD = np.asarray(VD, dtype=np.float64)
    GD = grad(VD, F)

    da = doublearea(VD, F) / 2
    MFD = sp.diags(da, 0, shape=(pre['f'], pre['f']), format='csr')

    LD = (GD.T @ sp.block_diag([MFD, MFD], format='csr') @ GD).tocsr()

    ne = EL.shape[0]
    au_LD = -np.asarray(LD[EL[:, 0], EL[:, 1]]).reshape(ne, 1)
    return au_LD
