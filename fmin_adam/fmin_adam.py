"""fmin_adam.m -- Dylan Muir's Adam optimiser with a MATLAB `fminunc` calling
convention.

fmin_adam - FUNCTION Adam optimiser, with matlab calling format

Usage: x, fval, exitflag, output = fmin_adam(fun, x0 [, stepSize, beta1, beta2,
                                             epsilon, nEpochSize, options])

'fmin_adam' is an implementation of the Adam optimisation algorithm (gradient
descent with Adaptive learning rates individually on each parameter, with
momentum) from [1]. Adam is designed to work on stochastic gradient descent
problems; i.e. when only small batches of data are used to estimate the gradient
on each iteration.

'fun' is a function handle `fCost [, vfCdX] = fun(x [, nIter])` defining the
function to minimise. It must return the cost at the parameter 'x', optionally
evaluated over a mini-batch of data. If analytical gradients are available
(recommended), then 'fun' must return the gradients in 'vfCdX', evaluated at 'x'
(optionally over a mini-batch). If analytical gradients are not available, then
complex-step finite difference estimates will be used.

To use analytical gradients (default), options['GradObj'] = 'on'. To force the
use of finite difference gradient estimates, options['GradObj'] = 'off'.

'fun' must be deterministic in its calculation of 'fCost' and 'vfCdX', even if
mini-batches are used. To this end, 'fun' can accept a parameter 'nIter' which
specifies the current iteration of the optimisation algorithm. 'fun' must return
estimates over identical problems for a given value of 'nIter'.

Steps that do not lead to a reduction in the function to be minimised are not
taken.

'exitflag' will be an integer value indicating why the algorithm terminated:
    0: An output or plot function indicated that the algorithm should terminate.
    1: The estimated reduction in 'fCost' was less than TolFun.
    2: The norm of the current step was less than TolX.
    3: The number of iterations exceeded MaxIter.
    4: The number of function evaluations exceeded MaxFunEvals.

'output' is a dict with .stepsize / .gradient / .funccount / .iteration / .fval /
.exitflag / .improvement.

References
[1] Diederik P. Kingma, Jimmy Ba. "Adam: A Method for Stochastic Optimization",
       ICLR 2015.

Author: Dylan Muir <dylan.muir@unibas.ch>
Created: 10th February, 2017

Porting notes
-------------
* `optimset`-style options become a plain dict, as core_optimize_block.py already
  does; missing keys fall back to the `'defaults'` struct, so a caller can pass
  just `{'MaxIter': 10, 'Display': 'iter'}`.
* MATLAB's `nargout(fun) < 2` guard cannot be reproduced (a Python callable does
  not advertise how many values it returns), so it is dropped; the solver's
  `value_grad_fun` returns `(value, grad, Hess, out_data)` and only the first two
  are read. `nargin(fun) < 2` is answered with `inspect.signature`.
* `xHist` is preallocated `numberofvariables x MaxIter+1` exactly as in the .m
  (which carries a commented-out `MappedTensor` for the same reason); with the
  default `MaxIter = 1e6` that will not fit, so a caller must set MaxIter, as
  core_variational_beltrami.py does.
* `x0` may be the solver's f-by-3 `da`; it is flattened column-major, and the
  returned `x` is the flat vector, matching fmin_vector_adam_simple.
* The complex-step gradient path is transcribed but needs a `fun` that accepts a
  complex `x`; the oracles here do not, so 'GradObj' must stay 'on'.
"""

import inspect

import numpy as np
import torch

from tqhm_config import DT, col


def fmin_adam(fun, x0=None, stepSize=None, beta1=None, beta2=None, epsilon=None,
              nEpochSize=None, options=None):
    # - Default parameters

    DEF_stepSize = 0.001
    DEF_beta1 = 0.9
    DEF_beta2 = 0.999
    DEF_epsilon = np.sqrt(np.finfo(np.float64).eps)

    # - Default options
    if isinstance(fun, str) and fun == 'defaults':
        return dict(Display='final',
                    GradObj='on',
                    DerivativeCheck='off',
                    MaxFunEvals=1e4,
                    MaxIter=1e6,
                    TolFun=1e-6,
                    TolX=1e-5,
                    UseParallel=False)

    # - Check arguments and assign defaults

    if x0 is None:
        raise TypeError('*** fmin_adam: Incorrect usage.')

    if stepSize is None:
        stepSize = DEF_stepSize

    if beta1 is None:
        beta1 = DEF_beta1

    if beta2 is None:
        beta2 = DEF_beta2

    if epsilon is None:
        epsilon = DEF_epsilon

    # optimset(@fmin_adam) is just the 'defaults' struct; a partial dict is
    # filled in from it rather than rejected.
    defaults = fmin_adam('defaults')
    defaults.update(OutputFcn=None, PlotFcns=None)
    if options is None or len(options) == 0:
        options = dict(defaults)
    else:
        options = {**defaults, **dict(options)}

    # - Parse options structure

    x0 = torch.as_tensor(x0)
    numberofvariables = x0.numel()

    # - Are analytical gradients provided?
    if options['GradObj'] == 'on':
        # - Check supplied cost function
        # (nargout(fun) cannot be inspected in Python; see the module docstring.)

        bUseAnalyticalGradients = True
        nEvalsPerIter = 2
    else:
        bUseAnalyticalGradients = False

        # - Wrap cost function for complex step gradients
        inner_fun = fun
        fun = lambda x, nIter: FA_FunComplexStepGrad(inner_fun, x, nIter)
        nEvalsPerIter = numberofvariables + 2

    # - Should we check analytical gradients?
    bCheckAnalyticalGradients = options['DerivativeCheck'] == 'on'

    # - Get iteration and termination options
    MaxIter = int(FA_eval(options['MaxIter'], numberofvariables))
    options['MaxIter'] = MaxIter
    options['MaxFunEvals'] = FA_eval(options['MaxFunEvals'], numberofvariables)

    # - Parallel operation is not yet implements
    if options['UseParallel']:
        print('--- fmin_adam: Warning: \'UseParallel\' is not yet implemented.')

    # - Check supplied function

    if _nargin(fun) < 2:
        # - Function does not make use of the 'nIter' argument, so make a wrapper
        inner_fun2 = fun
        fun = lambda x, nIter: inner_fun2(x)

    # - Check that gradients are identical for a given nIter
    if not bUseAnalyticalGradients:
        vfGrad0 = fun(x0, 1)[1]
        vfGrad1 = fun(x0, 1)[1]

        tol = np.spacing(float(torch.max(torch.maximum(
            torch.abs(vfGrad0), torch.abs(vfGrad1)))))
        if float(torch.max(torch.abs(vfGrad0 - vfGrad1))) > tol:
            raise ValueError(
                '*** fmin_adam: Cost function must return identical stochastic '
                'gradients for a given \'nIter\', when analytical gradients are '
                'not provided.')

    # - Check analytical gradients
    if bUseAnalyticalGradients and bCheckAnalyticalGradients:
        FA_CheckGradients(fun, x0)

    # - Check user function for errors
    try:
        fval0, vfCdX0 = fun(x0, 1)[:2]

    except Exception as mErr:
        raise RuntimeError(
            '*** fmin_adam: Error when evaluating function to minimise.') from mErr

    fval0 = float(fval0)
    vfCdX0 = col(vfCdX0)

    # - Check that initial point is reasonable
    if np.isinf(fval0) or np.isnan(fval0) or bool(torch.any(
            torch.isinf(vfCdX0) | torch.isnan(vfCdX0))):
        raise ValueError('*** fmin_adam: Invalid starting point for user '
                         'function. NaN or Inf returned.')

    # - Initialise algorithm

    # - Preallocate cost and parameters
    xHist = torch.zeros(numberofvariables, MaxIter + 1,
                        dtype=DT, device=x0.device)   # MappedTensor(...)
    xHist[:, 0] = col(x0)
    x = col(x0).clone()
    vfCost = np.zeros(MaxIter + 1)   # MATLAB zeros(1,MaxIter) then grows to +1

    if nEpochSize is None:
        nEpochSize = numberofvariables

    vfCost[0] = fval0
    fLastCost = fval0
    fval = fval0

    # - Initialise moment estimates
    m = torch.zeros(numberofvariables, dtype=DT, device=x.device)
    v = torch.zeros(numberofvariables, dtype=DT, device=x.device)

    # - Initialise optimization values
    optimValues = dict(fval=vfCost[0],
                       funccount=nEvalsPerIter,
                       gradient=vfCdX0,
                       iteration=0,
                       improvement=np.inf,
                       stepsize=0)

    # - Initial display
    FA_Display(options, x, optimValues, 'init', nEpochSize)
    FA_Display(options, x, optimValues, 'iter', nEpochSize)

    # - Initialise plot and output functions
    FA_CallOutputFunctions(options, x0, optimValues, 'init')
    FA_CallOutputFunctions(options, x0, optimValues, 'iter')

    # - Optimisation loop
    while True:
        # - Next iteration
        optimValues['iteration'] = optimValues['iteration'] + 1
        nIter = optimValues['iteration']

        # - Update biased 1st moment estimate
        m = beta1 * m + (1 - beta1) * col(optimValues['gradient'])
        # - Update biased 2nd raw moment estimate
        v = beta2 * v + (1 - beta2) * col(optimValues['gradient']) ** 2

        # - Compute bias-corrected 1st moment estimate
        mHat = m / (1 - beta1 ** nIter)
        # - Compute bias-corrected 2nd raw moment estimate
        vHat = v / (1 - beta2 ** nIter)

        # - Determine step to take at this iteration
        vfStep = stepSize * mHat / (torch.sqrt(vHat) + epsilon)

        # - Test step for true improvement, reject bad steps
        if float(fun(x - vfStep, nIter)[0]) <= fval:
            x = x - vfStep
            optimValues['stepsize'] = float(torch.max(torch.abs(vfStep)))

        # - Get next cost and gradient
        fval, optimValues['gradient'] = fun(x, nIter + 1)[:2]
        fval = float(fval)
        vfCost[nIter] = fval
        optimValues['funccount'] = optimValues['funccount'] + nEvalsPerIter

        # - Call display, output and plot functions
        bStop = FA_Display(options, x, optimValues, 'iter', nEpochSize)
        bStop = bStop | FA_CallOutputFunctions(options, x, optimValues, 'iter')

        # - Store historical x
        xHist[:, nIter] = x

        # - Update covergence variables
        # MATLAB's 1-based max(1, nIter+1-nEpochSize) start index.
        nFirstCost = max(0, nIter - nEpochSize)
        fEstCost = float(np.mean(vfCost[nFirstCost:nIter + 1]))
        fImprEst = abs(fEstCost - fLastCost)
        optimValues['improvement'] = fImprEst
        fLastCost = fEstCost
        optimValues['fval'] = fEstCost

        # - Check termination criteria
        if bStop:
            optimValues['exitflag'] = 0
            break

        if nIter > nEpochSize:
            if fImprEst < options['TolFun'] / nEpochSize:
                optimValues['exitflag'] = 1
                break

            if optimValues['stepsize'] < options['TolX']:
                optimValues['exitflag'] = 2
                break

            if nIter >= options['MaxIter'] - 1:
                optimValues['exitflag'] = 3
                break

            if optimValues['funccount'] > options['MaxFunEvals']:
                optimValues['exitflag'] = 4
                break

    # - Determine best solution
    vfCost = vfCost[:nIter + 1]
    # nBestParams = nanargmin(vfCost)
    nBestParams = int(np.argmin(vfCost))   # wangyu replaced with min.
    x = xHist[:, nBestParams]
    fval = float(vfCost[nBestParams])
    exitflag = optimValues['exitflag']
    output = optimValues

    # - Final display
    FA_Display(options, x, optimValues, 'done', nEpochSize)
    FA_CallOutputFunctions(options, x, optimValues, 'done')

    return x, fval, exitflag, output


# Utility functions

def _nargin(fun):
    """MATLAB nargin(fun) for a function handle."""
    try:
        params = inspect.signature(fun).parameters
    except (TypeError, ValueError):
        return 2
    n = 0
    for p in params.values():
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
            n += 1
        elif p.kind == p.VAR_POSITIONAL:
            return -1   # MATLAB's varargin sentinel
    return n


# FA_FunComplexStepGrad - FUNCTION Compute complex step finite difference
# gradient estimates for an analytial function
def FA_FunComplexStepGrad(fun, x, nIter):
    # - Step size
    fStep = np.sqrt(np.finfo(np.float64).eps)

    # - Get nominal value of function
    fVal = fun(x, nIter)
    if isinstance(fVal, tuple):
        fVal = fVal[0]

    # - Estimate gradients with complex step
    vfCdX = torch.zeros(x.numel(), dtype=DT, device=x.device)
    for nParam in range(x.numel() - 1, -1, -1):
        xStep = col(x).clone().to(torch.complex128)
        xStep[nParam] = xStep[nParam] + fStep * 1j
        r = fun(xStep, nIter)
        if isinstance(r, tuple):
            r = r[0]
        vfCdX[nParam] = torch.imag(torch.as_tensor(r)) / fStep

    return fVal, vfCdX


# FA_CheckGradients - FUNCTION Check that analytical gradients match finite
# difference estimates
def FA_CheckGradients(fun, x0):
    # - Get analytical gradients
    vfCdXAnalytical = col(fun(x0, 1)[1])

    # - Get complex-step finite-difference gradient estimates
    vfCdXFDE = FA_FunComplexStepGrad(fun, x0, 1)[1]

    print('--- fmin_adam: Checking analytical gradients...')

    # - Compare gradients
    vfGradDiff = torch.abs(vfCdXAnalytical - vfCdXFDE)
    fMaxDiff, nDiffIndex = torch.max(vfGradDiff, dim=0)
    fMaxDiff = float(fMaxDiff)
    fTolGrad = np.spacing(float(torch.max(torch.maximum(
        torch.abs(vfCdXFDE), torch.abs(vfCdXAnalytical)))))
    if fMaxDiff > fTolGrad:
        print('   Gradient check failed.')
        print('   Maximum difference between analytical and finite-step '
              'estimate: %.2g' % fMaxDiff)
        print('   Analytical: %.2g; Finite-step: %.2g'
              % (float(vfCdXAnalytical[nDiffIndex]), float(vfCdXFDE[nDiffIndex])))
        print('   Tolerance: %.2g' % fTolGrad)
        print('   Gradient indicies violating tolerance: [', end='')
        for idx in torch.nonzero(vfGradDiff > fTolGrad).reshape(-1).tolist():
            print('%d, ' % idx, end='')
        print(']')

        raise ValueError('*** fmin_adam: Gradient check failed.')

    print('   Gradient check passed. Well done!')


# FA_Display - FUNCTION Display the current state of the optimisation
# algorithm
def FA_Display(options, x, optimValues, state, nEpochSize):
    bStop = False

    # - Should we display anything?
    if options['Display'] == 'none':
        return bStop

    # - Determine what to display
    if state == 'init':
        if options['Display'] == 'iter':
            # The MATLAB template has only four %10s for five arguments, so the
            # format is recycled once for 'Step-size'; reproduced verbatim.
            print('\n\n%10s   %10s   %10s   %10s'
                  % ('Iteration', 'Func-count', 'f(x)', 'Improvement'))
            print('\n\n%10s   ' % 'Step-size', end='')
            print('%10s   %10s   %10s   %10s   %10s'
                  % ('----------', '----------', '----------',
                     '----------', '----------'))

    elif state == 'iter':
        if options['Display'] == 'iter' and (optimValues['iteration'] % nEpochSize) == 0:
            print('%10d   %10d   %10.2g   %10.2g   %10.2g'
                  % (optimValues['iteration'], optimValues['funccount'],
                     optimValues['fval'], optimValues['improvement'],
                     optimValues['stepsize']))

    elif state == 'done':
        bPrintSummary = (options['Display'] == 'final'
                         or options['Display'] == 'iter'
                         or (options['Display'] == 'notify'
                             and optimValues['exitflag'] != 1
                             and optimValues['exitflag'] != 2))

        if bPrintSummary:
            print('\n\n%10s   %10s   %10s   %10s   %10s'
                  % ('Iteration', 'Func-count', 'f(x)', 'Improvement',
                     'Step-size'))
            print('%10s   %10s   %10s   %10s   %10s'
                  % ('----------', '----------', '----------',
                     '----------', '----------'))
            print('%10d   %10d   %10.2g   %10.2g   %10.2g'
                  % (optimValues['iteration'], optimValues['funccount'],
                     optimValues['fval'], optimValues['improvement'],
                     optimValues['stepsize']))
            print('%10s   %10s   %10s   %10s   %10s'
                  % ('----------', '----------', '----------',
                     '----------', '----------'))

            strExitMessage = FA_GetExitMessage(optimValues, options)
            print('\nFinished optimization.\n   Reason: %s\n' % strExitMessage)

    return bStop


# FA_CallOutputFunctions - FUNCTION Call output and plot functions during
# optimisation
def FA_CallOutputFunctions(options, x, optimValues, state):
    bStop = False

    if options.get('OutputFcn') is not None:
        bStop = bStop | bool(options['OutputFcn'](x, optimValues, state))
        # drawnow

    if options.get('PlotFcns') is not None:
        if isinstance(options['PlotFcns'], (list, tuple)):
            bStop = bStop | any(bool(fh(x, optimValues, state))
                                for fh in options['PlotFcns'])
        else:
            bStop = bStop | bool(options['PlotFcns'](x, optimValues, state))
        # drawnow

    return bStop


# FA_eval - FUNCTION Evaluate a string or return a value
def FA_eval(oInput, numberofvariables=None):
    # MATLAB's evalin('caller', ...) resolves option strings such as
    # '100*numberofvariables' in fmin_adam's own workspace.
    if isinstance(oInput, str):
        return eval(oInput, {'numberofvariables': numberofvariables})
    return oInput


# FA_GetExitMessage - FUNCTION Return the message describing why the
# algorithm terminated
def FA_GetExitMessage(optimValues, options):
    if optimValues['exitflag'] == 0:
        return 'Terminated due to output or plot function.'

    elif optimValues['exitflag'] == 1:
        return ('Function improvement [%.2g] less than TolFun [%.2g].'
                % (optimValues['improvement'], options['TolFun']))

    elif optimValues['exitflag'] == 2:
        return ('Step size [%.2g] less than TolX [%.2g].'
                % (optimValues['stepsize'], options['TolX']))

    elif optimValues['exitflag'] == 3:
        return 'Number of iterations reached MaxIter [%d].' % options['MaxIter']

    elif optimValues['exitflag'] == 4:
        return ('Number of function evaluations reached MaxFunEvals [%d].'
                % options['MaxFunEvals'])

    else:
        return 'Unknown termination reason.'


# --- END of fmin_adam.py ---
