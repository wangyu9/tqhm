"""Compatibility shim over the libigl python bindings.

libigl renamed its I/O functions between releases: the OBJ reader is `read_obj`
in the 2.5.x bindings but `readOBJ` (matching the C++ name) in newer releases.
The geometry operators this repo uses (barycenter, edge_lengths, doublearea,
upsample, boundary_facets, ...) are snake_case in both, so only I/O needs a
shim. `igl_fn(name)` resolves a function by trying known aliases and returning
the first that exists, so call sites work unchanged across igl versions.
"""

import igl

# Lookup key is the name used at call sites; the value lists candidate attribute
# names to try in order (old snake_case first, then the newer C++ camelCase).
# Only the I/O functions this repo actually calls need aliasing.
_ALIASES = {
    'read_obj': ('read_obj', 'readOBJ'),
}


def igl_fn(name):
    """Return the igl function for `name`, trying known aliases across versions."""
    for cand in _ALIASES.get(name, (name,)):
        fn = getattr(igl, cand, None)
        if fn is not None:
            return fn
    tried = _ALIASES.get(name, (name,))
    raise AttributeError(
        'libigl has no function %r (tried %r); installed igl version: %s'
        % (name, tried, getattr(igl, '__version__', '?')))
