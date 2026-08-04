"""Fixed-pattern extraction of the Auu and Auk blocks of the Laplacian.

MATLAB slices A(unknown,unknown) and A(unknown,known) each iteration, which
re-sorts indices every time. Because the pattern never changes, the gather
indices are precomputed once here and each iteration is a pure gather.

This has no single .m counterpart: it replaces the index bookkeeping that
oracle_conjugate_newton_symmetric.m expresses via `q`/`Q` permutations and
SuiteSparse's symbolic factorization.
"""

import torch

from tqhm_config import DEV, IT, ti


def _lexsort_rc(r, c):
    """np.lexsort((c, r)): sort by r (major), then c (minor). Column-major-safe.

    argsort by the minor key first (c), then a stable argsort by the major key
    (r) — the verified torch equivalent of numpy's lexsort.
    """
    o1 = torch.argsort(c, stable=True)
    o = o1[torch.argsort(r[o1], stable=True)]
    return o


class LapBlocks:
    def __init__(self, n, indptr, indices, unknown, known):
        indptr = ti(indptr)
        indices = ti(indices)
        unknown = ti(unknown)
        known = ti(known)

        nu = int(unknown.numel())
        nk = int(known.numel())

        # map global vertex -> local index within unknown / known
        pos = torch.full((n,), -1, dtype=IT, device=DEV)
        pos[unknown] = torch.arange(nu, dtype=IT, device=DEV)
        is_unknown = torch.zeros(n, dtype=torch.bool, device=DEV)
        is_unknown[unknown] = True

        posk = torch.full((n,), -1, dtype=IT, device=DEV)
        posk[known] = torch.arange(nk, dtype=IT, device=DEV)

        rows = torch.repeat_interleave(torch.arange(n, dtype=IT, device=DEV),
                                       indptr[1:] - indptr[:-1])
        cols = indices
        entry = torch.arange(indices.numel(), dtype=IT, device=DEV)

        # --- Auu block ---
        m_uu = is_unknown[rows] & is_unknown[cols]
        r_uu, c_uu, e_uu = pos[rows[m_uu]], pos[cols[m_uu]], entry[m_uu]
        order = _lexsort_rc(r_uu, c_uu)
        r_uu, c_uu, e_uu = r_uu[order], c_uu[order], e_uu[order]

        counts = torch.bincount(r_uu + 1, minlength=nu + 1)
        indptr_uu = torch.cumsum(counts, 0).to(IT)

        self.nu = nu
        self.nk = nk
        self.indptr_uu = indptr_uu
        self.indices_uu = c_uu
        self.gather_uu = e_uu

        # --- Auk block ---
        is_known = torch.zeros(n, dtype=torch.bool, device=DEV)
        is_known[known] = True
        m_uk = is_unknown[rows] & is_known[cols]
        r_uk, c_uk, e_uk = pos[rows[m_uk]], posk[cols[m_uk]], entry[m_uk]
        self.Auk_shape = (nu, nk)
        self.Auk_rows = r_uk
        self.Auk_cols = c_uk
        self.gather_uk = e_uk

    def Auu_data(self, data):
        return data[self.gather_uu]

    def Auk_matvec(self, data, x_known):
        """Compute Auk @ x_known for a complex or real vector."""
        vals = data[self.gather_uk]
        contrib = vals.to(x_known.dtype) * x_known[self.Auk_cols]
        out = torch.zeros(self.nu, dtype=x_known.dtype, device=x_known.device)
        out.scatter_add_(0, self.Auk_rows, contrib)
        return out
