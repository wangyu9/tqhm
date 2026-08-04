"""val_grad_qhm.m -- (value, gradient) wrapper around QHWvalue.

`QHWvalue.m` does not exist anywhere in `../qhm` and is not on any path this port
can see, so the file is stale: only the packing/unpacking of `xx` is portable.
The call is kept as a lazy import so the code path survives, and raises
NotImplementedError with the reason when the dependency is genuinely missing.

Note MATLAB hard-codes `mcdim = 3` here (three tensor parameters per triangle),
unlike val_grad_oracle.m which derives the split from `numel(pre.BC)`.
"""

import numpy as np

from tqhm_config import npy


def val_grad_qhm(xx, pre, s_at2au, s_value_grad, fixed):
    try:
        from QHWvalue import QHWvalue
    except ImportError:
        raise NotImplementedError(
            'QHWvalue.m has no counterpart in ../qhm; val_grad_qhm.m is stale')

    B = pre.known

    assert pre.dim == 2
    mcdim = 3

    xx = npy(xx).reshape(-1, order='F')
    aatt = xx[0:pre.f * mcdim]
    BC = xx[pre.f * mcdim:]

    aatt = aatt.reshape((pre.f, aatt.size // pre.f), order='F')
    nb = np.asarray(B).ravel().size
    BC = BC.reshape((nb, BC.size // nb), order='F')

    e, g_at, g_BC, FW = QHWvalue(s_at2au(aatt), pre, aatt, BC, s_value_grad)

    # g_BC(1:size(fixed,1),:) = 0;

    g = np.concatenate([npy(g_at).reshape(-1, order='F'),
                        npy(g_BC).reshape(-1, order='F')])

    return e, g, FW
