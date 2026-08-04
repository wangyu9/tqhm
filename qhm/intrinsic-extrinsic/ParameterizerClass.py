"""ParameterizerClass.m -- proxy iterations plus line search for disc triangle
meshes, from the Accelerated Quadratic Proxy / Composite Majorization code base.

MATLAB `classdef ... < matlab.mixin.Copyable` with mutable state, so it is ported as
a normal Python class with instance methods of the same names. `Copyable` maps to
`copy.copy`, so no explicit `copy` method is added.

Two dependencies are outside this repository and are stubbed to raise:
`computeInjectiveStepSize` (a MEX kernel, see qhm/compileAllMex.m) and the
`TriangleMesh` class with `ComputeTutteParameterization` /
`computeSurfaceGradientMatrix` / `computeTriangleAreas`. EnergyClassSymDirichlet.Init
also needs `NewtonMex`, so nothing in this class actually runs end to end; it is
transcribed for completeness because the paper's solver never calls it.

MATLAB's `nargout` dispatch in ComputeEnergy becomes an explicit `nargout` argument
so that the three return shapes (f), (f,g) and (f,g,H) stay distinguishable.
"""

import numpy as np
import scipy.sparse as sp

from EnergyClassSymDirichlet import EnergyClassSymDirichlet


def computeInjectiveStepSize(F, x, p, tol):
    raise NotImplementedError(
        'computeInjectiveStepSize is a compiled MEX kernel from the AQP code base; '
        'not ported')


def TriangleMesh(*args, **kwargs):
    raise NotImplementedError(
        'the AQP TriangleMesh class is not ported (qhm/triangle_mesh.py is a '
        'different, struct-based mesh)')


class ParameterizerClass:

    EnergyTypesList = ['Sym-Dirichlet', 'Neo-Hookean']

    def __init__(self):
        self.M = None                    # triangle mesh class
        self.Energy = None               # Energy class
        self.ProxyTypesList = None
        self.ComputeProxy = None         # Proxy method from the Energy class
        # factor for schaefer stepsize for the first degenerated triangle
        self.c_linesearch = 0.9
        self.EnergyType = None
        # wangyu added to allow larger step size, 1.0 is the one used by the CM paper.
        self.t_max = 1.0

        # properties(SetAccess = private)
        self.Ac = None                   # linear constraints
        # convergence parameters
        # num of iterations in a row fulfilling the ftol or xtol criteria
        self.num_conv_iters = 5
        self.ftol = 1e-6
        self.xtol = 1e-10
        self.fcounter = None
        self.xcounter = None

        # bar angle
        self.bar_angle = 110

        self.ls_alpha = 0.2              # line search sufficient decrease
        self.ls_beta = 0.5               # line search step factor

        self.SetEnergyType('Sym-Dirichlet')

    def SetEnergyType(self, Type):
        self.EnergyType = Type
        if self.Energy is not None:
            # MATLAB calls the inherited handle `delete`; dropping the reference is
            # the Python equivalent.
            self.Energy = None
        if self.EnergyType == 'Sym-Dirichlet':
            self.ProxyTypesList = ['Composite Majorization', 'SLIM',
                                   'Projected Newton', 'Full projected Newton',
                                   'Newton', 'Temp']
            self.Energy = EnergyClassSymDirichlet()
        else:
            raise ValueError('incorrect energy type')
        self.SetProxyType(self.ProxyTypesList[0])

    def SetMesh(self, M):
        """performs initialize too"""
        self.M = M
        D1, D2 = self.M.computeSurfaceGradientMatrix()
        Ai = self.M.computeTriangleAreas()
        self.Energy.Init(D1, D2, Ai, self.M.V.T, self.M.F.T)
        K = self.Initialize(D1, D2, Ai)
        self.ResetOptimization()
        return K

    def ResetOptimization(self):
        self.fcounter = 0
        self.xcounter = 0

    def setAc(self):
        if False:
            # 3 coordinates
            s = [1, 1, 1]
            i = [0, 1, 2]
            j = [0, self.M.Nv, self.M.Nv + 1]

        # 1 point
        i = [0, 1]
        j = [0, self.M.Nv]
        s = [1, 1]

        if False:
            # all boundary
            Bi = np.concatenate(self.M.findOrientedBoundariesMatlab())
            s = np.ones(Bi.size * 2)
            i = np.arange(Bi.size * 2)
            j = np.r_[Bi, Bi + self.M.Nv]

        if False:
            # sum zero
            s = np.ones(self.M.Nv * 2)
            i = np.r_[np.zeros(self.M.Nv, dtype=int), np.ones(self.M.Nv, dtype=int)]
            j = np.r_[np.arange(self.M.Nv), np.arange(self.M.Nv) + self.M.Nv]

        i = np.asarray(i)
        j = np.asarray(j)
        s = np.asarray(s, dtype=np.float64)
        self.Ac = sp.coo_matrix((s, (i, j)), shape=(i.max() + 1, 2 * self.M.Nv)).tocsr()

    def SetProxyType(self, type):
        e = self.Energy
        if type == 'Composite Majorization':
            self.ComputeProxy = e.ProxyCompMajor
        elif type == 'SLIM':
            self.ComputeProxy = e.ProxySLIM
        elif type == 'Projected Newton':
            self.ComputeProxy = e.ProxyProjectedNewton
        elif type == 'Full projected Newton':
            self.ComputeProxy = e.ProxyProjectedNewtonFull
        elif type == 'Newton':
            self.ComputeProxy = e.ProxyNewton
        elif type == 'Temp':
            self.ComputeProxy = e.ProxyTemp

    def Initialize(self, D1, D2, Ai):
        if self.EnergyType == 'Sym-Dirichlet':
            # initialize with Tutte embedding into a circle
            K = TriangleMesh('VF', self.M.ComputeTutteParameterization(), self.M.F)
            self.setAc()
            f, g = self.ComputeEnergy(K.V.T.reshape(-1), nargout=2)
            # calculated scale for min sym-dirichlet energy
            fDirichlet, finvDirichlet = self.Energy.ComputeSymmetricDirichletParts()
            # (finvDirichlet/fDirichlet)**0.25
            scale = 1.0 / np.sum(self.M.computeTriangleAreas())
            K.V = scale * K.V
        else:
            raise ValueError('incorrect energy type')
        return K, f, g

    def DoIteration(self, x):
        """x is the current solution as [u,v] or the column stack [u; v];
        the returned x has the same shape as the input."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 2 and x.shape[1] == 2:
            x = x.ravel(order='F')
            reshapeFlag = True
        else:
            reshapeFlag = False

        # here x is the column stack of the parameterization
        f, g, H = self.ComputeEnergy(x, nargout=3)
        Nconstraints = self.Ac.shape[0]
        KKT = sp.bmat([[H, self.Ac.T],
                       [self.Ac, sp.csr_matrix((Nconstraints, Nconstraints))]],
                      format='csc')
        # condest(KKT)
        rhs = np.r_[-g, np.zeros(Nconstraints)]
        sol = sp.linalg.spsolve(KKT, rhs)
        p = sol[:2 * self.M.Nv]

        t, stepSize, f = self.DoLineSearch(x, p, f, g)
        x = x + t * p
        dx = np.linalg.norm(t * p)

        if reshapeFlag:
            x = x.reshape((-1, 2), order='F')

        return x, f, g, dx, stepSize

    def DoLineSearch(self, x, p, f, g):
        """x: current solution, p: step direction, f: current value, g: gradient.

        Returns (t, stepSize, f_new); stepSize is the step to the first degenerate
        triangle (capped at 1) after Smith & Schaefer.
        """
        stepSize = computeInjectiveStepSize(np.asarray(self.M.F).T, x, p, 1e-12)
        t = min(self.c_linesearch * stepSize, self.t_max)

        alpha_g_p = self.ls_alpha * (g @ p)

        def computeLineSearchCond():
            lhs = self.ComputeEnergy(x + t * p, nargout=1)
            rhs = f + t * alpha_g_p
            return lhs, rhs

        linesearch_cond_lhs, linesearch_cond_rhs = computeLineSearchCond()
        while linesearch_cond_lhs > linesearch_cond_rhs:
            t = self.ls_beta * t
            linesearch_cond_lhs, linesearch_cond_rhs = computeLineSearchCond()
        f_new = linesearch_cond_lhs

        return t, stepSize, f_new

    def ComputeEnergy(self, x, nargout=3):
        """f: value, g: gradient, H: the quadratic proxy Hessian used in the KKT."""
        self.Energy.UpdatePose(x)

        if nargout == 1:
            return self.Energy.ComputeEnergy(nargout=1)
        f, g = self.Energy.ComputeEnergy(nargout=2)
        if nargout > 2:
            H = self.ComputeProxy()
            return f, g, H
        return f, g

    def OptimizationConverged(self, fcur, fprev, xcur, xprev):
        status = 'Not Converged'
        if abs(fcur - fprev) < self.ftol * (fcur + 1):
            if self.fcounter >= self.num_conv_iters:
                return 'Change in energy < tol'
            self.fcounter = self.fcounter + 1
        else:
            self.fcounter = 0

        if np.linalg.norm(xcur - xprev) < self.xtol * (np.linalg.norm(xcur) + 1):
            if self.xcounter >= self.num_conv_iters:
                return 'Change in X < tol'
            self.xcounter = self.xcounter + 1
        else:
            self.xcounter = 0

        return status

    def delete(self):
        # MATLAB: clear mex
        pass
