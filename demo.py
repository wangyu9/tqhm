"""demo.m -- entry point.

MATLAB needs SuiteSparse on the path; here the sparse solve goes through tdss.py
(cuDSS), so there is no equivalent setup step.
"""

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


def main(meshname=None):
    if False:
        path = CASES / 'Simple'
        meshname = meshname or 'WeberZorin14_fig19'
        # 'cross', 'Lshape', 'WeberZorin14_fig19', 'square_transRot90',
        # 'square_rot180', 'WeberZorin14_fig20'
    else:
        path = CASES / 'Letters'
        meshname = meshname or 'david_o_A'   # 'david_o_A', 'bunny_i_H', 'lucy_o_G'

    if meshname in ('WeberZorin14_fig19', 'WeberZorin14_fig20', 'cross', 'Lshape',
                    'square_transRot90', 'square_rot180', 'square_rot90'):
        path = CASES / 'Simple'

    folder = path / meshname

    # --- set hyperparameters and options ---
    args = default_args()

    # set variable parameterization
    # 'complex-plane-det1-tanh' or 'complex-plane-det1' (sigmoid or tanh)
    args.tp_type = 'complex-plane-det1-tanh'
    args.solver = 'vadam'          # 'lbfgs' or 'vadam' (vector-adam)

    # lr schedule for adam:
    args.solver_adam_lr = np.r_[0.1 * np.ones(5), 0.01 * np.ones(5)]
    args.max_iter = args.solver_adam_lr.size

    args.vis = False
    args.load_format = 'du2020'
    args.alpha = 1                 # 100; 10000
    args.F = -1
    # 'none', 'symmetric-Dirichlet-capped', 'symmetric-Dirichlet', 'ARAP',
    # 'area-change', 'mass-spring'
    args.energy_type = 'none'

    # initial map: True is more robust numerically but less conformal;
    # False is less robust but more conformal.
    args.graph_tutte = True

    # --- run the solver ---
    out = core_variational_beltrami(folder, args)

    ut, vt = out['u'], out['v']

    if False:
        from render_mesh2 import render_mesh2
        render_mesh2(np.stack([ut.cpu().numpy(), vt.cpu().numpy()], axis=1),
                     out['F'], EdgeColor=[0, 0, 0], FaceColor=[1, 1, 1],
                     LineWidth=1)

    return out


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else None)
