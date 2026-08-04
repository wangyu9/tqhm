"""compileAllMex.m -- MEX build script from the AQP (Kovalsky & Galun 2016) code.

    Code implementing the paper "Accelerated Quadratic Proxy for Geometric
    Optimization", SIGGRAPH 2016.
    Written by Shahar Kovalsky and Meirav Galun.

It compiles computeMeshTranformationCoeffsMex, computeInjectiveStepSizeMex,
projectRotationMex, computeFunctionalIsoDistMex and projectRotationMexFast
against Eigen 3.4.0 / libigl. This port has no MEX layer -- the equivalent
numerics are torch/scipy -- so there is nothing to build.
"""


def compileAllMex():
    raise NotImplementedError(
        'compileAllMex.m builds MATLAB MEX binaries; this port has no MEX layer.')
