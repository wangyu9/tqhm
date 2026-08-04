"""tensor_para_inverse.m -- undo the radial squashing of tensor_para_forward.

`atanh` inverts the `tanh` of the 'complex-plane-det1-tanh' case, so this is the
inverse of tensor_para_forward for that para_type only -- the MATLAB source takes
no para_type argument and hard-codes atanh.  The third column of `at` (the
rotation/conformal-factor slot) is zero, i.e. the returned point is the canonical
preimage.

Note the input must satisfy `|au(:,1:2)| < 1` or atanh is not real; the MATLAB
source does not check either.  `f = size(au,1)` is used only for `zeros(f,1)`.
"""

import torch

from tqhm_config import DT


def tensor_para_inverse(au):
    # ff = lambda rr: 1 - 1 / (1 + torch.log(1 + rr))
    # gg = lambda rr: 1 / (1 + torch.log(1 + rr)) ** 2 / (1 + rr)
    #
    # ff = lambda rr: torch.tanh(rr)
    # gg = lambda rr: 1 - torch.tanh(rr) ** 2

    # case 'complex-plane-det1'
    #     ff = lambda rr: 1 - 1 / (1 + torch.log(1 + rr))
    #     gg = lambda rr: 1 / (1 + torch.log(1 + rr)) ** 2 / (1 + rr)
    #     res_sym_grad_complex_plane_det1
    # case 'complex-plane-det1-tanh'
    #     ff = lambda rr: torch.tanh(rr)
    #     gg = lambda rr: 1 - torch.tanh(rr) ** 2
    #     res_sym_grad_complex_plane_det1

    # s_at2au(at)
    # s_pdapdt_lmul = tp.s_pdapdt_lmul

    w1 = au[:, 0]
    w2 = au[:, 1]

    f = au.shape[0]

    # fr1 = lambda w1, w2: w1 / sqrt(w1**2+w2**2) * ff(sqrt(w1**2+w2**2))
    # fr2 = lambda w1, w2: w2 / sqrt(w1**2+w2**2) * ff(sqrt(w1**2+w2**2))

    w_rr_sq = w1 ** 2 + w2 ** 2
    w_rr = torch.sqrt(w_rr_sq)

    w_rr_aa = torch.atanh(w_rr)

    nw1 = w1 / w_rr
    nw2 = w2 / w_rr

    return torch.stack([w_rr_aa * nw1, w_rr_aa * nw2,
                        torch.zeros(f, dtype=DT, device=au.device)], dim=1)
