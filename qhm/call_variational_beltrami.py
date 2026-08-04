"""call_variational_beltrami.m -- thin wrapper that runs the main solver script.

In MATLAB the body is the bare statement `core_variational_beltrami;`, which
executes the script inside this function's workspace (so it sees `folder`/`args`
and leaves `out` behind). Here `core_variational_beltrami` is already a function,
so the wrapper just forwards and returns.
"""

from core_variational_beltrami import core_variational_beltrami


def call_variational_beltrami(folder, args):
    return core_variational_beltrami(folder, args)
