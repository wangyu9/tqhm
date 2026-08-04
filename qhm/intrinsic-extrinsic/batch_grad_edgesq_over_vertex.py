"""batch_grad_edgesq_over_vertex.m -- grad_edgesq_over_vertex for every triangle.

Hot path (called from IntrinsicHessianClass.GradsIntrinsic2Extrinsic), so this is
vectorized torch. Rows of the 6 axis are (x1,y1,x2,y2,x3,y3); columns of the 3 axis
are the squared lengths of the edges opposite vertices 1,2,3.
"""

import torch


def batch_grad_edgesq_over_vertex(VD, F):
    assert VD.shape[1] == 2

    f = F.shape[0]

    KTs = torch.zeros(f, 6, 3, dtype=VD.dtype, device=VD.device)

    v12 = VD[F[:, 1], :] - VD[F[:, 0], :]
    v23 = VD[F[:, 2], :] - VD[F[:, 1], :]
    v31 = VD[F[:, 0], :] - VD[F[:, 2], :]

    KTs[:, 0:2, 1] = 2 * v31
    KTs[:, 0:2, 2] = 2 * (-v12)

    KTs[:, 2:4, 2] = 2 * v12
    KTs[:, 2:4, 0] = 2 * (-v23)

    KTs[:, 4:6, 0] = 2 * v23
    KTs[:, 4:6, 1] = 2 * (-v31)

    # double check
    if False:
        import numpy as np
        from grad_edgesq_over_vertex import grad_edgesq_over_vertex

        VD_np = VD.detach().cpu().numpy()
        F_np = F.detach().cpu().numpy()
        KTs2 = np.zeros((f, 6, 3))

        for i in range(f):
            VTi = VD_np[F_np[i, :], :]
            KT = grad_edgesq_over_vertex(VTi)
            # KT: deodv, 6 x 3
            KTs2[i, :, :] = KT

        print(np.abs(KTs.detach().cpu().numpy() - KTs2).sum())

    return KTs
