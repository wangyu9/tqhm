"""readDuMatResV2.m -- readDuMatResV taking the vertex count from the header
instead of from the caller.

modified from Alec's readDMAT (see readDuMat.py for the original header).
"""

import numpy as np

from readDuMat import _read_tokens


def readDuMatResV2(filename):
    # open file
    tokens = _read_tokens(filename)

    # [nm,count] = fscanf(fp,'resV %d %d',[1 2])
    assert tokens[0] == 'resV'
    nm = [int(t) for t in tokens[1:3]]
    count = len(nm)
    tokens = tokens[3:]

    n = nm[0]
    m = nm[1]
    assert count == 2
    assert m == 2

    # read header
    size_W = n * 2
    # read data
    W = np.array([float(t) for t in tokens[:size_W]], dtype=np.float64)

    # close file

    if W.size != size_W:
        raise ValueError(
            'Size in header (%d) did not match size of data (%d) in file'
            % (size_W, W.size))

    # W = W.reshape((n, 2))

    return W.reshape((2, n), order='F').T
