"""intrinsic_grad_area.m -- intrinsic gradient with the triangle *angles*
regularized away from degeneracy.

Each triangle is replaced by a similar one built from clamped angles: F[:,0] at
the origin, F[:,1] at (cot(a1)+cot(a2), 0), F[:,2] at (cot(a1), 1). Because
`absorb_mass` is on, the frame is then scaled by 1/sqrt(Area) and the returned
Area is all ones, so the mass matrix is folded into G.

Returns (G, Area, GIS) matching `[G,Area,out]`; GIS carries g1/g2/g3 like
intrinsic_grad, plus the `absorb_mass` flag MATLAB stores in `out`.
"""

import numpy as np

from edge_lengths import edge_lengths
from doublearea_intrinsic import doublearea_intrinsic
from internalangles_intrinsic import internalangles_intrinsic
from intrinsic_grad import _core2d, _assemble


def intrinsic_grad_area(V, F, epsilon):
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F)

    absorb_mass = True

    out = {'absorb_mass': absorb_mass}

    n = V.shape[0]
    f = F.shape[0]

    E = edge_lengths(V, F)

    if False:
        E = E / E.max(axis=1, keepdims=True)

        # ll = np.flatnonzero(E < epsilon)
        # E.ravel(order='F')[ll] += epsilon

        ep1 = 0.6
        ep2 = 0.3

        tt = np.flatnonzero(E.min(axis=1) < ep1)
        # E[tt, :] += epsilon
        Et = E[tt, :]

        Et = np.abs(ep1 - Et.min(axis=1, keepdims=True)) + Et

        # tl = np.flatnonzero(Et < epsilon)
        # Et.ravel(order='F')[tl] += epsilon
        E[tt, :] = Et

        Area = doublearea_intrinsic(E) / 2

    # F[:,0] are always put at the origin.
    # F[:,1] are put at (e3, 0)
    # F[:,2] are stored at (e2 * cos(theta1), h3)
    # cos1 = (E[:,1]**2 + E[:,2]**2 - E[:,0]**2) / (2*E[:,1]*E[:,2])

    # V1 = np.zeros((f, 2))
    # V2 = np.stack([E[:, 2], np.zeros(f)], axis=1)
    # V3 = np.stack([(E[:,1]**2 + E[:,2]**2 - E[:,0]**2) / (2*E[:,2]),
    #                2*Area / E[:,2]], axis=1)

    AAA = internalangles_intrinsic(E) / np.pi

    if True:
        # the old version
        AAA[AAA < epsilon] = epsilon
    else:
        upper = max(0.0, 1 - 2 * epsilon)
        AAA[AAA > upper] = upper

        AAA[AAA < epsilon] = epsilon

    AAA = np.pi * AAA / AAA.sum(axis=1, keepdims=True)

    V1 = np.zeros((f, 2))
    V2 = np.stack([1 / np.tan(AAA[:, 0]) + 1 / np.tan(AAA[:, 1]), np.zeros(f)], axis=1)
    V3 = np.stack([1 / np.tan(AAA[:, 0]), np.ones(f)], axis=1)

    Area = (1 / np.tan(AAA[:, 0]) + 1 / np.tan(AAA[:, 1])) / 2

    assert Area.min() > 0

    if absorb_mass:
        s = np.sqrt(Area)[:, None]
        V1 = V1 / s
        V2 = V2 / s
        V3 = V3 / s
        Area = np.ones_like(Area)

    from tqhm_config import td
    g1 = _core2d(td(V1), td(V2), td(V3))[:, :2]
    g2 = _core2d(td(V2), td(V3), td(V1))[:, :2]
    g3 = _core2d(td(V3), td(V1), td(V2))[:, :2]

    G = _assemble(g1, g2, g3, F, n, f)

    out['g1'] = g1
    out['g2'] = g2
    out['g3'] = g3

    return G, Area, out
