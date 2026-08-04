"""writeOBJ_safe.m -- write an OBJ with vertex/texture/normal information.

  writeOBJ_safe(filename, V, F, UV, TF, N, NF)

  Input:
   filename  path to .obj file
   V  #V by 3 list of vertices
   F  #F by 3 list of triangle indices
   UV  #UV by 2 list of texture coordinates
   TF  #TF by 3 list of corner texture indices into UV
   N  #N by 3 list of normals
   NF  #NF by 3 list of corner normal indices into N

Not delegated to `igl.write_obj`: that writer uses its own float formatting and
refuses the mixed / partially-missing corner-index cases this one handles, so the
`%g` output and the per-face branch structure are transcribed directly.

Indices in this port are 0-based (readOBJ goes through igl), while the OBJ format
is 1-based, so F/TF/NF get +1 on the way out and MATLAB's `<= 0` test for a
missing corner index becomes `< 0`.
"""

import warnings

import numpy as np

from tqhm_config import npy


def writeOBJ_safe(filename, V, F, UV=None, TF=None, N=None, NF=None):
    # print('writing: ' + filename)
    V = np.asarray(npy(V), dtype=np.float64)
    F = np.asarray(npy(F), dtype=np.int64)

    if V.shape[1] == 2:
        warnings.warn('Appending 0s as z-coordinate')
        V = np.c_[V, np.zeros(V.shape[0])]
    else:
        assert V.shape[1] == 3

    hasN = N is not None and np.size(N) > 0
    hasUV = UV is not None and np.size(UV) > 0

    if hasUV:
        UV = np.asarray(npy(UV), dtype=np.float64)
    if hasN:
        N = np.asarray(npy(N), dtype=np.float64)

    with open(filename, 'w') as f:
        for row in V:
            f.write('v %g %g %g\n' % (row[0], row[1], row[2]))

        if hasUV:
            if UV.shape[1] == 2:
                for row in UV:
                    f.write('vt %g %g\n' % (row[0], row[1]))
            elif UV.shape[1] == 3:
                for row in UV:
                    f.write('vt %g %g %g\n' % (row[0], row[1], row[2]))

        if hasN:
            for row in N:
                f.write('vn %g %g %g\n' % (row[0], row[1], row[2]))

        if hasUV and (TF is None or np.size(TF) == 0):
            TF = F
        if hasN and (NF is None or np.size(NF) == 0):
            NF = F

        if TF is not None:
            TF = np.asarray(npy(TF), dtype=np.int64)
        if NF is not None:
            NF = np.asarray(npy(NF), dtype=np.int64)

        Fo = F + 1
        TFo = None if TF is None else TF + 1
        NFo = None if NF is None else NF + 1

        if not hasN and not hasUV:
            # A lot faster if we just have faces and they're all triangles
            fmt = 'f' + ' %d' * F.shape[1] + '\n'
            for row in Fo:
                f.write(fmt % tuple(row))
        else:
            for k in range(F.shape[0]):
                if ((not hasN) and (not hasUV)) \
                        or (np.any(TF[k, :] < 0) and np.any(NF[k, :] < 0)):
                    fmt = 'f' + ' %d' * F.shape[1] + '\n'
                    f.write(fmt % tuple(Fo[k, :]))
                elif hasUV and ((not hasN) or np.any(NF[k, :] < 0)):
                    fmt = 'f' + ' %d/%d' * F.shape[1] + '\n'
                    # MATLAB fprintf on the 2 x #F matrix [F(k,:);TF(k,:)]
                    # consumes it column-major: v1/vt1 v2/vt2 v3/vt3
                    f.write(fmt % tuple(np.stack([Fo[k, :], TFo[k, :]], axis=0)
                                        .ravel(order='F')))
                elif hasN and ((not hasUV) or np.any(TF[k, :] < 0)):
                    fmt = 'f' + ' %d//%d' * F.shape[1] + '\n'
                    # MATLAB has [F(k,:);TF(k,:)]' here -- transposed, and TF
                    # rather than NF. Column-major that flattens to
                    # (F1,F2,F3,TF1,TF2,TF3), i.e. "f F1//F2 F3//TF1 TF2//TF3".
                    # Preserved as written; this branch only fires for a
                    # normals-only mesh with a partly invalid TF.
                    f.write(fmt % tuple(np.stack([Fo[k, :], TFo[k, :]], axis=0)
                                        .T.ravel(order='F')))
                elif hasN and hasUV:
                    assert np.all(NF[k, :] >= 0)
                    assert np.all(TF[k, :] >= 0)
                    fmt = 'f' + ' %d/%d/%d' * F.shape[1] + '\n'
                    f.write(fmt % tuple(
                        np.stack([Fo[k, :], TFo[k, :], NFo[k, :]], axis=0)
                        .ravel(order='F')))
