"""find_files_with_ext.m -- the .obj files in a folder.

The MATLAB name is a leftover: the extension is hard-coded to '.obj', not a
parameter. `files(3:end)` there skips '.' and '..'; pathlib.iterdir never yields
them, so nothing is dropped here.

MATLAB's `dir` returns entries in the OS order (usually but not reliably sorted);
this returns them sorted by name so the output is deterministic.
"""

from pathlib import Path


def find_files_with_ext(folder):
    files = sorted(Path(folder).iterdir(), key=lambda p: p.name)
    names = [p.name for p in files]
    isMesh = [False] * len(names)

    for k in range(len(names)):
        ext = Path(names[k]).suffix
        if ext == '.obj':
            isMesh[k] = True

    return [nm for nm, keep in zip(names, isMesh) if keep]
