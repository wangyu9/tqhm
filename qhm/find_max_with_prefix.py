"""find_max_with_prefix.m -- highest-numbered file matching <prefix><id><ext>.

Returns (pathname, fullname, id). `id` is -1 when nothing matches, and the
returned name is then literally '<prefix>-1<ext>' -- the MATLAB code does not
guard against that, and callers use the id to detect it.

MATLAB's `str2num` returns [] for a non-numeric middle part, and `max(id, [])`
leaves `id` unchanged; the try/except below reproduces that.
"""

from pathlib import Path


def find_max_with_prefix(prefix, folder, ext):
    len_ = len(prefix)

    len_ext = len(ext)

    # startIndex = regexp('result_Du_orgiter_95','result_Du_orgiter_\d*')

    files = sorted(Path(folder).iterdir(), key=lambda p: p.name)

    id_ = -1

    for p in files:
        name = p.name
        if len(name) > (len_ + len_ext) and name[:len_] == prefix:
            try:
                current = int(float(name[len_:len(name) - len_ext]))
            except ValueError:
                continue   # MATLAB: str2num -> [], max(id,[]) == id
            id_ = max(id_, current)

    fullname = prefix + str(id_) + ext

    pathname = str(folder) + '/' + fullname
    return pathname, fullname, id_
