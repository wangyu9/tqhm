"""initialize.m -- path/renderer setup from the AQP (Kovalsky & Galun 2016) code.

    Code implementing the paper "Accelerated Quadratic Proxy for Geometric
    Optimization", SIGGRAPH 2016.
    Disclaimer: The code is provided as-is for academic use only and without any
                guarantees. Please contact the author to report any bugs.
    Written by Shahar Kovalsky (http://www.wisdom.weizmann.ac.il/~shaharko/)
               Meirav Galun (http://www.wisdom.weizmann.ac.il/~/meirav/)

The MATLAB body is `addpath(genpath(...))` for toolbox/code/mex plus OpenGL
renderer tweaks. `demo.py` handles sys.path and matplotlib needs no renderer
selection, so the only faithful part left is the path append; the `global
path_def` guard becomes a module-level flag.
"""

import sys
import time
from pathlib import Path

_path_def = False


def initialize(root='.'):
    global _path_def

    # reset timer;
    t0 = time.time()
    dummy = time.time() - t0

    # set paths
    if not _path_def:
        print('- Adding toolbox paths')

        # common path
        root = Path(root).resolve()
        for sub in ['toolbox', 'code', 'mex']:
            p = root / sub
            if p.is_dir() and str(p) not in sys.path:
                sys.path.insert(0, str(p))

        # MATLAB disabled OpenGL and forced the zbuffer renderer on unix; there
        # is no matplotlib equivalent worth reproducing.

        _path_def = True
