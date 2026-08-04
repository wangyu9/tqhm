"""example_variational_beltrami.m -- same wrapper as call_variational_beltrami.m,
with a usage example kept in an `if false` block.

The example block calls this very function recursively and points at the
Locally-Injective-Mappings-Benchmark, which is not part of this repo; the path
logic is kept as written and would fail at call time.
"""

import numpy as np

from default_args import default_args
from core_variational_beltrami import core_variational_beltrami


def example_variational_beltrami(folder, args):
    if False:
        # --- example of use ---
        name = "Simple/square_rot180"
        # "square_transRot90" "Simple/square_rot180"; "Simple/WeberZorin14_fig19"
        # "Simple/square_rot90" # "Simple/cross"
        # name = "Letters/gargoyle_i_A"; # "Letters/hand_3_i_A";

        folder = "../Locally-Injective-Mappings-Benchmark/" + name + "/"
        # folder = "D:/WorkSpace/fastsol/Locally-Injective-Mappings-Benchmark/Simple/" + name + "/"

        # MATLAB: addpath(genpath('./'));  demo.py already sets up sys.path.
        args = default_args()

        # "Letters/david_o_A"; # "Letters/lucy_o_G"; # "Letters/bunny_i_G";

        # args.mprint = @(fid, varargin)

        # log_path = folder + '/200x50-lbfgs-'
        # diary(log_path+'log.txt')

        out = example_variational_beltrami(folder, args)

        # save(log_path+'out.mat')

        u = out['u']
        v = out['v']
        F = out['F']
        from render_mesh2 import render_mesh2
        render_mesh2(np.stack([u, v], axis=1), F,
                     EdgeColor=[0, 0, 0], FaceColor=[1, 1, 1])
        # axis off; axis equal;

        # diary off

    return core_variational_beltrami(folder, args)
