"""fmin_vector_adam_simple.m -- Vector Adam (arXiv:2205.13599).

Differs from plain Adam in the second moment: v is a per-row scalar accumulated
from the squared norm of the whole row's gradient, so the update is rotation
equivariant within each row.
"""

import numpy as np
import torch

from tqhm_config import DEV, DT, get_verbose


def fmin_vector_adam_simple(fun, x0, stepSize, beta1, beta2, options, d, args=None):
    epsilon = np.sqrt(np.finfo(np.float64).eps)   # MATLAB sqrt(eps)

    MaxIter = options['MaxIter']
    if args is None:
        args = {'break_with_stop': False}

    x = x0.reshape(d, -1).t().contiguous() if x0.dim() == 1 else x0.clone()
    if x0.dim() == 1:
        # MATLAB reshape(x0,[numel/d, d]) is column-major
        x = x0.reshape(d, -1).t().contiguous()

    n = x.shape[0]

    m = torch.zeros(n, d, dtype=DT, device=DEV)
    v = torch.zeros(n, dtype=DT, device=DEV)

    for it in range(1, MaxIter + 1):
        value, grad, _, out_data = fun(_col(x))

        if get_verbose() >= 2:
            print('vadam: iter=%04d, f=%g' % (it, float(value)))

        if out_data['stop_sign'] and args.get('break_with_stop', False):
            print('VAdam: stop_sign set.')
            break

        grad = grad.reshape(d, -1).t().contiguous() if grad.dim() == 1 else grad

        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * torch.sum(grad ** 2, dim=1)

        mt = m / (1 - beta1 ** it)
        vt = v / (1 - beta2 ** it)

        x = x - stepSize * mt / (torch.sqrt(vt)[:, None] + epsilon)

    return _col(x)


def _col(x):
    """MATLAB x(:) for a 2D tensor: column-major flatten."""
    return x.t().reshape(-1).contiguous()
