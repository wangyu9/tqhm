"""tensor_para_faster.m -- map the free variables `at` to the metric tensor `au`
and provide the chain-rule left-multiply d(au)/d(at).

The variable is a point w=(w1,w2) in the plane; it is squashed into the unit
disc by f(r)*w/r (tanh or the log variant), then mapped to a unit-determinant
symmetric positive tensor. w3 (at[:,2]) is unused, matching the MATLAB code
where the third column of the gradient is identically zero.
"""

import torch

from tqhm_config import DT


def tensor_para_faster(at, para_type):
    if at.dim() == 1:
        at = at.reshape(3, -1).t()

    w1 = at[:, 0]
    w2 = at[:, 1]
    f = at.shape[0]

    w_rr_sq = w1 ** 2 + w2 ** 2
    w_rr = torch.sqrt(w_rr_sq)

    nw1 = w1 / w_rr
    nw2 = w2 / w_rr

    if para_type == 'complex-plane-det1':
        fff = 1 - 1 / (1 + torch.log1p(w_rr))
        ggg = 1 / (1 + torch.log1p(w_rr)) ** 2 / (1 + w_rr)
    elif para_type == 'complex-plane-det1-tanh':
        th = torch.tanh(w_rr)
        fff = th
        ggg = 1 - th ** 2
    else:
        raise ValueError('Error: Unimplemented!')

    pfr11 = fff * (nw2 ** 2 / w_rr) + ggg * nw1 ** 2
    pfr12 = fff * (-nw1 * nw2 / w_rr) + ggg * nw1 * nw2
    pfr21 = pfr12
    pfr22 = fff * (nw1 ** 2 / w_rr) + ggg * nw2 ** 2

    p1 = nw1 * fff
    p2 = nw2 * fff

    rr_sq = p1 ** 2 + p2 ** 2
    D = rr_sq - 1.0

    mA1 = -((p1 - 1.0) ** 2 + p2 ** 2) / D
    mA2 = (p2 * 2.0) / D
    mA3 = -((p1 + 1.0) ** 2 + p2 ** 2) / D

    au = torch.stack([mA1, mA2, mA3], dim=1)

    ps_dAdP11 = -(p1 * 2.0 - 2.0) / D + p1 * ((p1 - 1.0) ** 2 + p2 ** 2) / D ** 2 * 2.0
    ps_dAdP12 = (p2 * -2.0) / D + p2 * ((p1 - 1.0) ** 2 + p2 ** 2) / D ** 2 * 2.0

    ps_dAdP21 = p1 * p2 / D ** 2 * -4.0
    ps_dAdP22 = 2.0 / D - p2 ** 2 / D ** 2 * 4.0

    ps_dAdP31 = -(p1 * 2.0 + 2.0) / D + p1 * ((p1 + 1.0) ** 2 + p2 ** 2) / D ** 2 * 2.0
    ps_dAdP32 = (p2 * -2.0) / D + p2 * ((p1 + 1.0) ** 2 + p2 ** 2) / D ** 2 * 2.0

    zero = torch.zeros(f, dtype=DT, device=at.device)

    s_dAdP1 = [ps_dAdP11 * pfr11 + ps_dAdP12 * pfr21,
               ps_dAdP11 * pfr12 + ps_dAdP12 * pfr22,
               zero]
    s_dAdP2 = [ps_dAdP21 * pfr11 + ps_dAdP22 * pfr21,
               ps_dAdP21 * pfr12 + ps_dAdP22 * pfr22,
               zero]
    s_dAdP3 = [ps_dAdP31 * pfr11 + ps_dAdP32 * pfr21,
               ps_dAdP31 * pfr12 + ps_dAdP32 * pfr22,
               zero]

    def s_pdapdt(gg, Area):
        """eval_grad: rg(:,ii) = Area .* sum_j s_j{ii} .* gg(:,j)."""
        gg = gg.reshape(3, -1).t() if gg.dim() == 1 else gg
        out = []
        for ii in range(3):
            out.append(Area * (s_dAdP1[ii] * gg[:, 0]
                               + s_dAdP2[ii] * gg[:, 1]
                               + s_dAdP3[ii] * gg[:, 2]))
        return torch.cat(out)

    paupat = {'s_dAdP1': s_dAdP1, 's_dAdP2': s_dAdP2, 's_dAdP3': s_dAdP3,
              's_pdapdt': s_pdapdt}
    return au, paupat
