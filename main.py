"""main.py -- command-line front end for core_variational_beltrami.

Same job as demo.py, but every field of the args struct that can be expressed on
a command line is exposed through argparse. Fields that are not CLI-friendly
(the reg_value/reg_grad closures, the solver_adam_args dict, fine_recorder) keep
their default_args() values.

Running `python main.py` with no flags reproduces demo.main('david_o_A').
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for p in [ROOT, ROOT / 'qhm', ROOT / 'gptoolbox']:
    sys.path.insert(0, str(p))

import numpy as np

from default_args import default_args
from core_variational_beltrami import core_variational_beltrami

# the test-case meshes are vendored into this repo so it is self-contained
CASES = ROOT / 'test_cases'

# meshes that live under test_cases/Simple/ rather than test_cases/Letters/
SIMPLE_MESHES = ('WeberZorin14_fig19', 'WeberZorin14_fig20', 'cross', 'Lshape',
                 'square_transRot90', 'square_rot180', 'square_rot90')


def _resolve_folder(mesh, folder, cases):
    if folder is not None:
        return Path(folder)
    base = cases / ('Simple' if mesh in SIMPLE_MESHES else 'Letters')
    return base / mesh


def build_parser():
    d = default_args()
    p = argparse.ArgumentParser(
        description='Run the variational Beltrami solver on a mesh.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # --- mesh selection ---
    p.add_argument('mesh', nargs='?', default='david_o_A',
                   help='mesh name under test_cases/{Simple,Letters}/')
    p.add_argument('--folder', default=None,
                   help='explicit path to the mesh folder (overrides --mesh)')
    p.add_argument('--cases', default=str(CASES),
                   help='root directory holding the test_cases subfolders')

    # --- solver ---
    p.add_argument('--solver', choices=['lbfgs', 'vadam'], default='vadam',
                   help='inner optimizer')
    p.add_argument('--solver-adam-lr', type=float, nargs='+',
                   default=[0.1] * 5 + [0.01] * 5,
                   help='adam/vadam learning rate; one value or a per-iter schedule')
    p.add_argument('--solver-lbfgs-m', type=int, default=d.solver_lbfgs_m,
                   help='L-BFGS memory (maxcor)')
    p.add_argument('--max-iter', type=int, default=None,
                   help='outer iterations (default: length of --solver-adam-lr)')

    # --- parameterization / energy ---
    p.add_argument('--tp-type', default='complex-plane-det1-tanh',
                   help="tensor parameterization, e.g. 'complex-plane-det1-tanh'")
    p.add_argument('--energy-type', default='none',
                   help="nonlinear energy: 'none', 'symmetric-Dirichlet', 'ARAP', ...")
    p.add_argument('--alpha', type=float, default=1.0)
    p.add_argument('--beta', type=float, default=d.beta)
    p.add_argument('--gamma', type=float, default=d.gamma)
    p.add_argument('--L2-reg', type=float, default=d.L2_reg)
    p.add_argument('--F', type=int, default=-1,
                   help='faster routine when face count <= this (-1 always uses it)')

    # --- loading / preprocessing ---
    p.add_argument('--load-format',
                   choices=['du2020', 'du2020-old', 'du2020-proj', 'none'],
                   default='du2020')
    p.add_argument('--load-normalize-target',
                   action=argparse.BooleanOptionalAction,
                   default=d.load_normalize_target)
    p.add_argument('--sub-div-level', type=int, default=d.sub_div_level)
    p.add_argument('--rotate', type=float, default=d.rotate)
    p.add_argument('--epsilon-gradarea-angle', type=float,
                   default=d.epsilon_gradarea_angle,
                   help='regularize triangle angles away from degeneracy')

    # --- initialization / termination ---
    p.add_argument('--graph-tutte', action=argparse.BooleanOptionalAction,
                   default=True,
                   help='initialize from the uniform graph Laplacian')
    p.add_argument('--min-iter-attempt-term', type=int,
                   default=d.min_iter_attempt_term)
    p.add_argument('--stop-when-no-flip', action=argparse.BooleanOptionalAction,
                   default=d.stop_when_no_flip)

    # --- output ---
    p.add_argument('--verbose', type=int, default=1,
                   help='0 silent; >=1 prints the per-iteration flip line; '
                        '>=2 also prints the per-inner-iteration energy line')
    p.add_argument('--vis', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--save-file', default=d.save_file,
                   help='path prefix for saving results (npz)')
    return p


def args_from_namespace(ns):
    """Start from default_args() and override the CLI-representable fields."""
    args = default_args()

    args.solver = ns.solver
    lr = np.atleast_1d(np.asarray(ns.solver_adam_lr, dtype=np.float64))
    args.solver_adam_lr = lr if lr.size > 1 else float(lr[0])
    args.solver_lbfgs_m = ns.solver_lbfgs_m
    args.max_iter = ns.max_iter if ns.max_iter is not None else int(lr.size)

    args.tp_type = ns.tp_type
    args.energy_type = ns.energy_type
    args.alpha = ns.alpha
    args.beta = ns.beta
    args.gamma = ns.gamma
    args.L2_reg = ns.L2_reg
    args.F = ns.F

    args.load_format = ns.load_format
    args.load_normalize_target = ns.load_normalize_target
    args.sub_div_level = ns.sub_div_level
    args.rotate = ns.rotate
    args.epsilon_gradarea_angle = ns.epsilon_gradarea_angle

    args.graph_tutte = ns.graph_tutte
    args.min_iter_attempt_term = ns.min_iter_attempt_term
    args.stop_when_no_flip = ns.stop_when_no_flip

    args.vis = ns.vis
    args.save_file = ns.save_file
    return args


def main(argv=None):
    ns = build_parser().parse_args(argv)

    from tqhm_config import set_verbose
    set_verbose(ns.verbose)

    folder = _resolve_folder(ns.mesh, ns.folder, Path(ns.cases))
    args = args_from_namespace(ns)

    out = core_variational_beltrami(folder, args)

    ut, vt = out['u'], out['v']

    if args.vis:
        from render_mesh2 import render_mesh2
        render_mesh2(np.stack([ut.cpu().numpy(), vt.cpu().numpy()], axis=1),
                     out['F'], EdgeColor=[0, 0, 0], FaceColor=[1, 1, 1],
                     LineWidth=1)

    return out


if __name__ == '__main__':
    main()
