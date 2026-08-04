"""attempt_local_global.m -- a local/global fallback used when only a handful of
triangles remain flipped.

MATLAB script; here a function. Only the `old_imp == false` branch is live: the
`old_imp == true` branch is kept for reference but needs the explicit GI operator
and a dense-column solve, which this port never builds.

Returns (u, v, num_flipped, a11, a12, a22).
"""

import numpy as np
import torch

from tqhm_config import DEV, DT
from tensor_para_faster import tensor_para_faster


def attempt_local_global(mesh, da, tp, reuse, BC, Area, f):
    old_imp = False

    V = mesh['V']
    known = mesh['IKB']
    unknown = mesh['IUB']

    if old_imp:
        au = tp['s_at2au'](da.reshape(3, -1).t().contiguous()).reshape(3, f).t()
    else:
        au, _ = tensor_para_faster(da.reshape(3, -1).t().contiguous(), tp['para_type'])
        au = au * Area[:, None]

    a11 = au[:, 0]
    a12 = au[:, 1]
    a22 = au[:, 2]

    u = v = None
    num_flipped = f

    for jj in range(1, 21):
        # --- global step ---
        print('attempting local-global step\n')

        if old_imp:
            raise NotImplementedError(
                'old_imp branch of attempt_local_global.m is not ported')

        uvc = torch.complex(V[:, 0].clone(), V[:, 1].clone())
        uvc[known] = torch.complex(BC[:, 0], BC[:, 1])

        grad_xy = reuse.grad_xy

        data = reuse.RL.asb_full(mesh['GIS'], a11, a12, a22)
        Auu_data = reuse.sub.Auu_data(data)
        rhs = -reuse.sub.Auk_matvec(data, uvc[known])

        uvc[unknown] = reuse.solver.solve_complex(Auu_data, rhs)

        u = uvc.real.contiguous()
        v = uvc.imag.contiguous()

        if bool(torch.isnan(u).any()) or bool(torch.isnan(v).any()):
            print('Warning: local-global: UV has NaN\n')
            break

        # --- local step ---
        Gx_uvc, Gy_uvc = grad_xy(uvc)

        Gxu = Gx_uvc.real
        Gxv = Gx_uvc.imag
        Gyu = Gy_uvc.real
        Gyv = Gy_uvc.imag

        ccc = Area / torch.abs(Gxu * Gyv - Gxv * Gyu)

        a22 = (Gxu * Gxu + Gxv * Gxv) * ccc
        a12 = -(Gxu * Gyu + Gxv * Gyv) * ccc
        a11 = (Gyu * Gyu + Gyv * Gyv) * ccc

        newArea = Area * (Gxu * Gyv - Gxv * Gyu)

        flipped = newArea < 0
        num_flipped = int(torch.count_nonzero(flipped).item())
        print('flipps=%d, min_area=%g' % (num_flipped, float(newArea.min())), end='')

        if num_flipped == 0:
            break

    return u, v, num_flipped, a11, a12, a22
