"""Global device/dtype configuration and small MATLAB-compatibility helpers.

The MATLAB code runs in double precision; tdss.py (cuDSS) requires float64 on
CUDA, so the whole pipeline uses torch.float64 on the CUDA device.
"""

import numpy as np
import torch

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DT = torch.float64
IT = torch.int64

# Console verbosity. 1 prints only the per-outer-iteration flip line; 2 (default)
# also prints the per-inner-iteration energy line. Defaults to 2 so demo.py and
# validate.py see the full log they parse; main.py overrides via --verbose.
_VERBOSE = 2


def set_verbose(level):
    global _VERBOSE
    _VERBOSE = int(level)


def get_verbose():
    return _VERBOSE


def td(x):
    """To float64 tensor on the configured device."""
    if torch.is_tensor(x):
        return x.to(device=DEV, dtype=DT)
    return torch.as_tensor(np.asarray(x, dtype=np.float64), device=DEV, dtype=DT)


def tc(x):
    """To complex128 tensor on the configured device."""
    if torch.is_tensor(x):
        return x.to(device=DEV, dtype=torch.complex128)
    return torch.as_tensor(np.asarray(x), device=DEV, dtype=torch.complex128)


def ti(x):
    """To int64 tensor on the configured device."""
    if torch.is_tensor(x):
        return x.to(device=DEV, dtype=IT)
    return torch.as_tensor(np.asarray(x), device=DEV, dtype=IT)


def npy(x):
    """To a numpy array."""
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def col(x):
    """MATLAB x(:) -- flatten in column-major order."""
    if torch.is_tensor(x):
        return x.t().reshape(-1) if x.dim() == 2 else x.reshape(-1)
    x = np.asarray(x)
    return x.reshape(-1, order='F')


def reshape_f(x, shape):
    """MATLAB reshape(x, shape) -- column-major."""
    if torch.is_tensor(x):
        flat = col(x)
        if len(shape) == 2:
            return flat.reshape(shape[1], shape[0]).t().contiguous()
        return flat.reshape(shape)
    return np.asarray(x).reshape(shape, order='F')


def find_column_major(M):
    """MATLAB [I,J,V] = find(sparse M): nonzeros ordered column-major.

    Returns (I, J, V) as numpy arrays.
    """
    import scipy.sparse as sp
    M = sp.coo_matrix(M)
    order = np.lexsort((M.row, M.col))
    return M.row[order], M.col[order], M.data[order]
