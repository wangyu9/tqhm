"""default_args2.m -- default_args() with the vector-adam schedule used for the
harder benchmark meshes."""

import numpy as np

from default_args import default_args


def default_args2():
    args = default_args()

    args.sub_div_level = 0
    args.energy_type = 'none'
    args.vis = True

    args.F = 0   # do not use faster routine.

    args.min_iter_attempt_term = -1

    # args.energy_type = 'area'

    args.solver = 'vadam'
    # args.solver = 'lbfgs'

    args.solver_adam_lr = np.r_[0.1 * np.ones(15), 0.01 * np.ones(8),
                                0.001 * np.ones(2)]
    # args.solver_adam_lr = 10.0 ** np.linspace(-1, -3, 10)
    # args.solver_adam_lr = 0.1 * np.ones(5)

    args.max_iter = args.solver_adam_lr.size

    args.tp_type = 'complex-plane-det1-tanh'
    return args
