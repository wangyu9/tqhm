"""core_optimize_block.m -- one outer iteration: run the inner solver on `da`,
then re-evaluate to get the map and the flip count.

MATLAB script sharing the caller's workspace; here a function taking/returning
the pieces it touches. `ii` is 1-based to match the MATLAB printouts.

The 'lbfgs' branch calls fmincon with a limited-memory BFGS Hessian
approximation; scipy's L-BFGS-B is the equivalent here (`solver_lbfgs_m` maps to
`maxcor`, and MATLAB's OptimalityTolerance/StepTolerance to ftol/gtol).
"""

import numpy as np
import torch

from tqhm_config import DT, npy, get_verbose
from fmin_vector_adam_simple import fmin_vector_adam_simple


def core_optimize_block(ii, da, value_grad_fun, args, f, history,
                        LB=None, UB=None, NONLCON=None):
    if args.solver == 'lbfgs':
        from scipy.optimize import minimize

        shape = da.shape

        def fg(x):
            xt = torch.as_tensor(x, dtype=DT, device=da.device).reshape(shape)
            value, grad, _, _ = value_grad_fun(xt)
            return float(value), npy(grad).astype(np.float64).ravel()

        res = minimize(fg, npy(da).ravel(), jac=True, method='L-BFGS-B',
                       bounds=None if LB is None else list(zip(LB, UB)),
                       options={'maxiter': 50, 'maxcor': args.solver_lbfgs_m,
                                'ftol': 0.0, 'gtol': 1e-18, 'disp': True})
        da = torch.as_tensor(res.x, dtype=DT, device=da.device).reshape(shape)

    elif args.solver == 'vadam':
        sOpt = {'MaxIter': 50}

        lr = np.atleast_1d(np.asarray(args.solver_adam_lr, dtype=np.float64))
        if lr.size == 1:
            adam_lr = float(lr[0])
        else:
            assert lr.size == args.max_iter
            adam_lr = float(lr[ii - 1])

        if get_verbose() >= 2:
            print('\t adam_lr=%g' % adam_lr, end='')

        da = fmin_vector_adam_simple(value_grad_fun, da, adam_lr, 0.9, 0.999,
                                     sOpt, da.numel() // f, args.solver_adam_args)
    else:
        raise ValueError('undefined solver type!\n')

    _, _, _, out_data = value_grad_fun(da)

    u = out_data['u']
    v = out_data['v']

    # number of flipped triangles
    num_flipped = out_data['num_flipped']
    if get_verbose() >= 1:
        sep = '\n' if get_verbose() >= 2 else ''
        print('%sIter %04d, flipps %04d' % (sep, ii, num_flipped))

    record = {'u': u, 'v': v, 'num_flipped': num_flipped, 'da': da}
    history.append(record)

    if args.vis:
        from render_mesh2 import render_mesh2
        render_mesh2(torch.stack([u, v], dim=1), None,
                     EdgeColor=[0, 0, 0], FaceColor=[1, 1, 1])

    return da, u, v, num_flipped
