"""mesh_covariance.m -- per-vertex-star covariance matrices of a map u.

COV is the 2n x 2n assembly of the small 3x3 covariance blocks (the JJ/JJ term
is included so the result stays PSD); COVD keeps only the two diagonal blocks.
`u` is the stacked [x; y] vector of length 2n.
"""

import numpy as np
import scipy.sparse as sp


def mesh_covariance(n, F, u):
    F = np.asarray(F, dtype=np.int64)
    u = np.asarray(u, dtype=np.float64).ravel(order='F')

    II = np.concatenate([F[:, 0], F[:, 1], F[:, 2]])
    JJ = np.concatenate([F[:, 1], F[:, 2], F[:, 0]])

    PL = sp.coo_matrix((np.ones(II.size), (II, JJ)), shape=(n, n)).tocsr()
    degree = np.asarray((PL + PL.T).sum(axis=1)).ravel()
    # an interior edge is counted twice, a boundary edge only once, so degree/2
    # is the number of adjacent triangles per vertex
    _ = degree

    # indices assembling each small 3x3 covariance matrix
    MI = np.concatenate([II, II, JJ, JJ])   # the last JJ is needed, or it is not psd
    MJ = np.concatenate([II, JJ, II, JJ])

    covII = np.concatenate([MI, MI, MI + n, MI + n])
    covJJ = np.concatenate([MJ, MJ + n, MJ, MJ + n])

    covDII = np.concatenate([MI, MI + n])
    covDJJ = np.concatenate([MJ, MJ + n])

    COV = sp.coo_matrix((u[covII] * u[covJJ], (covII, covJJ)),
                        shape=(2 * n, 2 * n)).tocsr()

    COVD = sp.coo_matrix((u[covDII] * u[covDJJ], (covDII, covDJJ)),
                         shape=(2 * n, 2 * n)).tocsr()

    return COV, COVD
