"""boundary_faces.m -- boundary facets of a simplicial complex."""

import numpy as np
import igl


def boundary_faces(F):
    F = np.asarray(F, dtype=np.int32)
    return np.asarray(igl.boundary_facets(F), dtype=np.int64)
