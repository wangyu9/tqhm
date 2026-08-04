"""default_args.m"""


class Args(dict):
    """dict with attribute access, so args.foo and args['foo'] both work."""

    __getattr__ = dict.__getitem__

    def __setattr__(self, k, v):
        self[k] = v


def default_args():
    args = Args()
    args.solver = 'lbfgs'          # other choice is 'vadam'

    args.solver_adam_lr = 0.01     # the default lr if using adam or vadam
    args.solver_lbfgs_m = 10

    args.solver_adam_args = {'break_with_stop': False}
    # whether to stop when out_data.stop_sign is true

    args.sub_div_level = 0
    args.energy_type = 'none'
    args.energy_Wf = None

    args.vis = False
    args.max_iter = 20
    args.rotate = 0

    args.alpha = 1
    args.beta = 0
    args.gamma = 0

    args.F = 10
    # less equal than the number invokes a faster routine

    args.load_format = 'du2020'
    args.load_normalize_target = True

    args.graph_tutte = False

    args.save_file = None

    args.reg_value = lambda at, Area: 0
    args.reg_grad = lambda at, Area: 0

    args.tp_type = 'complex-plane-det1'

    args.min_iter_attempt_term = 16

    args.fine_recorder = None

    args.L2_reg = None

    args.stop_when_no_flip = True

    args.epsilon_gradarea_angle = None

    args.indEC = None
    return args
