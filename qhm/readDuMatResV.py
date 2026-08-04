"""readDuMatResV.m -- readDuMat for files carrying a "resV n m" header line.

modified from Alec's readDMAT (see readDuMat.py for the original header).

The MATLAB source consumes the header with `fscanf(fp,'resV %d %d',[1 2])` and
then ignores it, still trusting the caller's `n`; that is preserved -- see
readDuMatResV2 for the version that uses the header.
"""

import numpy as np

from readDuMat import _read_tokens


def readDuMatResV(filename, n):
    # open file
    tokens = _read_tokens(filename)

    # fscanf(fp,'resV %d %d',[1 2]) -- consumed and discarded
    assert tokens[0] == 'resV'
    tokens = tokens[3:]

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
