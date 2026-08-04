"""fmin_adam_simple.m -- plain Adam, no option parsing, no step rejection.

MATLAB takes `size(x,1)` for the state length, i.e. x0 must be a column vector.
A 2-D input is flattened column-major here so the solver's `da` (f-by-3) can be
passed directly, as `fmin_vector_adam_simple` allows.

`fun` returns (value, grad, Hess, out_data) like the rest of the solver; only the
first two are used.
"""

import numpy as np
import torch

from tqhm_config import DT, col


def fmin_adam_simple(fun, x0, stepSize, beta1, beta2, options):
    epsilon = np.sqrt(np.finfo(np.float64).eps)   # MATLAB sqrt(eps)

    MaxIter = options['MaxIter']

    x = col(x0).clone()

    n = x.shape[0]

    m = torch.zeros(n, dtype=DT, device=x.device)
    v = torch.zeros(n, dtype=DT, device=x.device)

    for it in range(1, MaxIter + 1):

        value, grad = fun(x)[:2]

        grad = col(grad)

        m = beta1 * m + (1 - beta1) * grad

        v = beta2 * v + (1 - beta2) * grad ** 2

        mt = m / (1 - beta1 ** it)
        vt = v / (1 - beta2 ** it)

        x = x - stepSize * mt / (torch.sqrt(vt) + epsilon)

    return x
