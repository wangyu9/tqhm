"""intrinsic_grad_hessian.m -- symbolic energy density, gradient and Hessian in
terms of the squared edge lengths of the rest (aa0,bb0,cc0) and deformed
(aa1,bb1,cc1) triangle.

MATLAB uses the Symbolic Toolbox plus `matlabFunction`; here sympy plus `lambdify`.
`matlabFunction` orders its arguments alphabetically, which for these six symbols
is (aa0, aa1, bb0, bb1, cc0, cc1) -- the order every call site uses -- and that
signature is reproduced exactly, including for energies whose derivatives no longer
depend on all six.

The returned closures are elementwise and broadcast over torch tensors, so a whole
mesh goes through in one call:

    e(a0,a1,b0,b1,c0,c1) -> (f,)
    g(a0,a1,b0,b1,c0,c1) -> (3,f)
    h(a0,a1,b0,b1,c0,c1) -> (3,3,f)
"""

import numpy as np
import sympy as sp
import torch

_SYMS = sp.symbols('aa0 aa1 bb0 bb1 cc0 cc1', real=True)
_ARGS = _SYMS   # matlabFunction's alphabetical order


def _sqrt(x):
    return torch.sqrt(x) if torch.is_tensor(x) else np.sqrt(x)


def _abs(x):
    return torch.abs(x) if torch.is_tensor(x) else np.abs(x)


def _sign(x):
    return torch.sign(x) if torch.is_tensor(x) else np.sign(x)


# torch tensors on CUDA do not go through numpy ufuncs, so the elementary
# functions sympy emits are routed to torch explicitly.
_MODULES = [{'sqrt': _sqrt, 'Abs': _abs, 'sign': _sign}, 'numpy']


def _symbolic_energy(energy):
    aa0, aa1, bb0, bb1, cc0, cc1 = _SYMS

    R = sp.sqrt(2 * aa0 * bb0 + 2 * aa0 * cc0 + 2 * bb0 * cc0
                - aa0 ** 2 - bb0 ** 2 - cc0 ** 2)

    A = ((-aa0 + bb0 + cc0) * aa1 + (aa0 - bb0 + cc0) * bb1
         + (aa0 + bb0 - cc0) * cc1) / R ** 2

    ysq = ((bb0 * cc0) * aa1 ** 2 + (aa0 * cc0) * bb1 ** 2 + (aa0 * bb0) * cc1 ** 2
           + (-aa0 - bb0 + cc0) * cc0 * bb1 * aa1
           + (-aa0 + bb0 - cc0) * bb0 * aa1 * cc1
           + (aa0 - bb0 - cc0) * aa0 * cc1 * bb1
           ) / R ** 4

    # singular values
    S = [sp.sqrt(A + 2 * sp.sqrt(ysq)), sp.sqrt(A - 2 * sp.sqrt(ysq))]

    if energy == 'area':
        # https://en.wikipedia.org/wiki/Heron%27s_formula
        e = sp.Rational(1, 4) * sp.sqrt(2 * (aa1 * bb1 + bb1 * cc1 + cc1 * aa1)
                                        - (aa1 ** 2 + bb1 ** 2 + cc1 ** 2)) / \
            (sp.Rational(1, 4) * sp.sqrt(2 * (aa0 * bb0 + bb0 * cc0 + cc0 * aa0)
                                         - (aa0 ** 2 + bb0 ** 2 + cc0 ** 2)))
    elif energy == 'area-r':
        epsilon = 1e-18
        e = sp.Rational(1, 4) * sp.sqrt(epsilon + 2 * (aa1 * bb1 + bb1 * cc1 + cc1 * aa1)
                                        - (aa1 ** 2 + bb1 ** 2 + cc1 ** 2)) / \
            (sp.Rational(1, 4) * sp.sqrt(2 * (aa0 * bb0 + bb0 * cc0 + cc0 * aa0)
                                         - (aa0 ** 2 + bb0 ** 2 + cc0 ** 2)))
    elif energy == 'mass-spring':
        e = 1e-3 * ((sp.sqrt(aa1) - sp.sqrt(aa0)) ** 2
                    + (sp.sqrt(bb1) - sp.sqrt(bb0)) ** 2
                    + (sp.sqrt(cc1) - sp.sqrt(cc0)) ** 2) / (aa0 + bb0 + cc0)
    elif energy == 'symmetric-Dirichlet':
        e = (A * 2 + A * 2 / (A ** 2 - 4 * ysq)) / 2
    elif energy == 'symmetric-Dirichlet-capped':
        epsilon = 1e-6
        e = 1e-6 * (A * 2 + A * 2 / (A ** 2 - 4 * ysq + epsilon)) / 2
    elif energy == 'symmetric-Dirichlet-equiv':
        # sym Dirichlet in an equivalent form.
        e = (S[0] ** 2 + S[1] ** 2 + 1 / (S[0] ** 2) + 1 / (S[1] ** 2)) / 2
    elif energy == 'ARAP':
        e = 1e-3 * (2 * A + 2 - 2 * (S[0] + S[1]))
    elif energy == 'ARAP-equiv':
        e = 1e-3 * ((S[0] - 1) ** 2 + (S[1] - 1) ** 2)
    elif energy == 'area-change':
        e = 1e-3 * (1 - sp.sqrt(2 * (aa1 * bb1 + bb1 * cc1 + cc1 * aa1)
                                - (aa1 ** 2 + bb1 ** 2 + cc1 ** 2)) /
                    sp.sqrt(2 * (aa0 * bb0 + bb0 * cc0 + cc0 * aa0)
                            - (aa0 ** 2 + bb0 ** 2 + cc0 ** 2))) ** 2
    else:
        raise ValueError('unsupport energy')

    return e


def _broadcast_like(x, ref):
    """A constant derivative lambdifies to a python scalar; expand it."""
    if torch.is_tensor(x):
        return x if x.shape == ref.shape else x.expand_as(ref)
    return torch.full_like(ref, float(x))


def intrinsic_grad_hessian(energy):
    e = _symbolic_energy(str(energy))

    aa1, bb1, cc1 = _SYMS[1], _SYMS[3], _SYMS[5]
    dvars = (aa1, bb1, cc1)

    G = [sp.diff(e, v) for v in dvars]
    H2 = [[sp.diff(e, vi, vj) for vj in dvars] for vi in dvars]

    e_raw = sp.lambdify(_ARGS, e, modules=_MODULES, cse=True)
    g_raw = sp.lambdify(_ARGS, G, modules=_MODULES, cse=True)
    h_raw = sp.lambdify(_ARGS, H2, modules=_MODULES, cse=True)

    def e_fun(*args):
        return e_raw(*args)

    def g_fun(*args):
        gg = g_raw(*args)
        ref = next((x for x in gg if torch.is_tensor(x)), None)
        if ref is None:
            return torch.as_tensor([float(x) for x in gg], dtype=torch.float64)
        return torch.stack([_broadcast_like(x, ref) for x in gg], dim=0)

    def h_fun(*args):
        hh = h_raw(*args)
        flat = [x for row in hh for x in row]
        ref = next((x for x in flat if torch.is_tensor(x)), None)
        if ref is None:
            return torch.as_tensor([[float(x) for x in row] for row in hh],
                                   dtype=torch.float64)
        return torch.stack([torch.stack([_broadcast_like(x, ref) for x in row], dim=0)
                            for row in hh], dim=0)

    return e_fun, g_fun, h_fun
