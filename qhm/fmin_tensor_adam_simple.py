"""fmin_tensor_adam_simple.m -- Vector Adam (arXiv:2205.13599) with a tensor norm.

Same as fmin_vector_adam_simple except the per-row second moment weighs the
middle (off-diagonal) component twice: sum_fun(gg) = gg1^2 + 2*gg2^2 + gg3^2,
which is the Frobenius norm of the symmetric 2x2 gradient. It also lacks the
printing and the `stop_sign` early exit of the vector variant.
"""

import numpy as np
import torch

from tqhm_config import DT


def fmin_tensor_adam_simple(fun, x0, stepSize, beta1, beta2, options, d):
    epsilon = np.sqrt(np.finfo(np.float64).eps)   # MATLAB sqrt(eps)

    MaxIter = options['MaxIter']

    # MATLAB reshape(x0,[numel(x0)/d, d]) is column-major
    x0 = x0.reshape(d, -1).t().contiguous() if x0.dim() == 1 else x0

    x = x0.clone()

    n = x.shape[0]

    m = torch.zeros(n, d, dtype=DT, device=x.device)
    v = torch.zeros(n, dtype=DT, device=x.device)

    sum_fun = lambda gg: gg[:, 0] ** 2 + 2 * gg[:, 1] ** 2 + gg[:, 2] ** 2

    for it in range(1, MaxIter + 1):

        value, grad = fun(_col(x))[:2]

        grad = grad.reshape(d, -1).t().contiguous() if grad.dim() == 1 else grad

        m = beta1 * m + (1 - beta1) * grad

        v = beta2 * v + (1 - beta2) * sum_fun(grad)

        mt = m / (1 - beta1 ** it)
        vt = v / (1 - beta2 ** it)

        x = x - stepSize * mt / (torch.sqrt(vt)[:, None] + epsilon)

    return _col(x)


def _col(x):
    """MATLAB x(:) for a 2D tensor: column-major flatten."""
    return x.t().reshape(-1).contiguous()
