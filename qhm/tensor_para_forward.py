"""tensor_para_forward.m -- map the free variables back to a Beltrami coefficient.

The radial squashing of the complex-plane parameterizations, applied on its own:
`au(:,1:2)` is read as a point of the complex plane, its modulus `w_rr` is passed
through `fff` and the direction `nw = w / |w|` is kept, so the result has modulus
`fff(|w|) < 1`.  This is the `ff` of tensor_para.m / res_sym_grad_complex_plane_det1
without the tensor assembly or any derivative.

`f = size(au,1)` is computed and never used in the MATLAB source; it is dropped
here.  The big comment block at the top of the .m (the ff/gg pairs of each
para_type) is kept because it documents which `fff` goes with which case.
"""

import torch


def tensor_para_forward(au, para_type):
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

    if para_type == 'complex-plane-det1':
        fff = lambda w_rr: 1 - 1 / (1 + torch.log(1 + w_rr))

    elif para_type == 'complex-plane-det1-tanh':
        fff = lambda w_rr: torch.tanh(w_rr)

    else:
        raise NotImplementedError('Error: Unimplemented!\n')

    w1 = au[:, 0]
    w2 = au[:, 1]

    # fr1 = lambda w1, w2: w1 / sqrt(w1**2+w2**2) * ff(sqrt(w1**2+w2**2))
    # fr2 = lambda w1, w2: w2 / sqrt(w1**2+w2**2) * ff(sqrt(w1**2+w2**2))

    w_rr_sq = w1 ** 2 + w2 ** 2
    w_rr = torch.sqrt(w_rr_sq)

    w_rr_aa = fff(w_rr)

    nw1 = w1 / w_rr
    nw2 = w2 / w_rr

    return torch.stack([w_rr_aa * nw1, w_rr_aa * nw2], dim=1)
