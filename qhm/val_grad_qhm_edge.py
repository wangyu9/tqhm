"""val_grad_qhm_edge.m -- (value, gradient) wrapper around FEMvalue_ew.

Identical to val_grad_oracle.m except that the number of tensor variables comes
from `size(pre.star,1)` (one weight per edge) rather than from `numel(pre.BC)`.
No solve happens here; FEMvalue_ew.py documents the solver.
"""

import numpy as np

from tqhm_config import npy


def _opt(option, name, default=None):
    if isinstance(option, dict):
        return option.get(name, default)
    return getattr(option, name, default)


def val_grad_qhm_edge(xx, pre, s_at2au, s_value_grad, option):
    from FEMvalue_ew import FEMvalue_ew

    B = pre.known

    assert pre.dim == 2

    ne = pre.star.shape[0]

    xx = npy(xx).reshape(-1, order='F')
    if _opt(option, 'at_only'):
        aatt = xx
        BC = npy(_opt(option, 'BC'))
    else:
        aatt = xx[0:ne]
        BC = xx[ne:]

    nb = np.asarray(B).ravel().size
    BC = np.asarray(BC).reshape((nb, np.size(BC) // nb), order='F')

    e, g_at, g_BC, FW, _ = FEMvalue_ew(s_at2au(aatt), pre, aatt, BC,
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
