"""val_grad_oracle.m -- thin (value, gradient) wrapper around FEMoracle.

`xx` packs the tensor variables `at` and (unless `option.at_only`) the boundary
positions `BC`. No solve happens here; FEMoracle.py documents the solver.
"""

import numpy as np

from tqhm_config import npy


def _opt(option, name, default=None):
    if isinstance(option, dict):
        return option.get(name, default)
    return getattr(option, name, default)


def val_grad_oracle(xx, pre, s_at2au, s_value_grad, option):
    from FEMoracle import FEMoracle

    B = pre.known

    assert pre.dim == 2

    xx = npy(xx).reshape(-1, order='F')
    ne = xx.size - np.size(pre.BC)

    if _opt(option, 'at_only'):
        aatt = xx
        BC = npy(_opt(option, 'BC'))
    else:
        aatt = xx[0:ne]
        BC = xx[ne:]

    nb = np.asarray(B).ravel().size
    BC = np.asarray(BC).reshape((nb, np.size(BC) // nb), order='F')

    e, g_at, g_BC, FW, _ = FEMoracle(s_at2au(aatt), pre, aatt, BC,
                                     s_value_grad, _opt(option, 'CR'))

    if _opt(option, 'PBC') is not None:
        g_BC = _opt(option, 'PBC')(g_BC)

    # g_BC(1:size(option.fixed,1),:) = 0;
    g_BC = npy(g_BC).reshape(-1, order='F')

    if _opt(option, 'PA') is not None:
        g_at = _opt(option, 'PA')(g_at)

    if _opt(option, 'at_only'):
        g = g_at
    else:
        g = np.concatenate([npy(g_at).reshape(-1, order='F'), g_BC])

    return e, g, FW
