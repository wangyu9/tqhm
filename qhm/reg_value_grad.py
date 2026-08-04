"""reg_value_grad.m -- the radial regularizer on the free variables `at`.

Penalizes the modulus r = |at(:,1:2)| by `weight * 4 * (cosh(r)^2 - 1)`, which is
`weight * 2 * (cosh(2r) - 1)`, i.e. flat to second order at r = 0 and blowing up
as the Beltrami coefficient approaches the unit circle (recall the
complex-plane-det1-tanh parameterization sends r through tanh).  `AI` is the
per-face integration weight (the mass matrix diagonal).

`reg_type` is accepted and never read, exactly as in the MATLAB source.

Two things to note about the gradient:
  * `reshape([g .* AI, 0*at(:,3)], [numel(at),1])` is column-major, so the
    returned vector is [d/dat1; d/dat2; 0] stacked by column -- `col` here.
  * the `weight` factor appears in `energy_radical` *and* again in front of the
    reshape, while `grad_radical` has no `weight`; so value and gradient are
    consistent (grad_radical is the derivative of energy_radical/weight).
  * at r = 0 the gradient is 0/0 in the MATLAB source too (the `./ r`); no guard
    is added.
"""

import torch

from tqhm_config import col


def _grad_radical(at):
    # only at(:,0:2) will be used.

    r = torch.sqrt(at[:, 0] ** 2 + at[:, 1] ** 2)

    g = (4 * 2) * torch.cosh(r) * torch.sinh(r) \
        * torch.stack([at[:, 0], at[:, 1]], dim=1) / r[:, None]

    return g


def reg_value_grad(reg_type, weight):
    # weight = 1e-3
    # weight = 1e-7
    energy_radical = lambda at: weight * 4 * (
        torch.cosh(torch.sqrt(at[:, 0] ** 2 + at[:, 1] ** 2)) ** 2 - 1)

    reg_value = lambda at, AI: torch.sum(AI * energy_radical(at))

    reg_grad = lambda at, AI: weight * col(
        torch.cat([_grad_radical(at) * AI[:, None], (0 * at[:, 2])[:, None]], dim=1))

    return reg_value, reg_grad
