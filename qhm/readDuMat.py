"""readDuMat.m -- read the per-iteration vertex dump of [Du et al. 2020].

modified from Alec's readDMAT.

READDMAT  read a matrix from a dmat file.  first line is <# columns> <# rows>,
then values with columns running faster

  W = readDMAT(filename)

  Input:
    filename  name of .dmat file
  Output:
    W  matrix read from file

  See also: writeDMAT

These files are whitespace-separated ASCII, *not* MATLAB .mat: the MATLAB source
reads them with `fscanf('%g')`, so this is a plain tokenizer, not scipy.io.
There is no header -- the caller supplies `n` -- and the 2n values arrive with
the two coordinates running fastest, hence reshape([2,n])'.
"""

import numpy as np


def _read_tokens(filename):
    """MATLAB fscanf's whitespace-delimited token stream."""
    with open(filename, 'r') as fp:
        return fp.read().split()


def readDuMat(filename, n):
    # open file
    tokens = _read_tokens(filename)
    # read header
    size_W = n * 2
    # read data
    W = np.array([float(t) for t in tokens[:size_W]], dtype=np.float64)

    # The commented-out binary-fallback branch of readDMAT is not reproduced;
    # it asserted the ASCII read had returned nothing, which never happens here.

    # close file

    # size should match header
    if W.size != size_W:
        raise ValueError(
            'Size in header (%d) did not match size of data (%d) in file'
            % (size_W, W.size))

    # W = W.reshape((n, 2))

    return W.reshape((2, n), order='F').T
