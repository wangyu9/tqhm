"""toy_rod_model.m -- standalone sandbox: a spring-chain rod minimized either over
the interior node positions or over the element lengths.

MATLAB script (no `function` line), so this is a function taking no workspace.

MATLAB builds the energy symbolically (`sym`, `jacobian`, `matlabFunction`) and
hands `fmincon` a {value, grad} pair. There is nothing symbolic to preserve here:
the energy is a sum of squared spring extensions, so the gradient is written out
explicitly (PORTING.md: no autograd, transcribe the derivative). `fmincon` with
box bounds becomes scipy's L-BFGS-B, as in core_optimize_block.py.

MATLAB's `rand` sequence is not reproduced, so the random initializations -- and
hence the local minima the 10 experiments land in -- differ run to run.
"""

import numpy as np


def _cal_length(x1, x2):
    return np.sqrt((x1[0] - x2[0]) ** 2 + (x1[1] - x2[1]) ** 2)


def _cal_energy(x_input, num_elem, stiffness, elem_leng, boundary, using_len=False):
    x_input = np.asarray(x_input, dtype=np.float64)

    if using_len:
        return float(np.sum(0.5 * stiffness * (x_input.ravel() - elem_leng) ** 2))

    rod_pos_tmp = np.concatenate([boundary[0:1, :],
                                 x_input.reshape((num_elem - 1, 2)),
                                 boundary[1:2, :]], axis=0)
    energy = 0.0
    for i in range(num_elem):
        energy += 0.5 * stiffness * (
            _cal_length(rod_pos_tmp[i, :], rod_pos_tmp[i + 1, :]) - elem_leng) ** 2
    return float(energy)


def _cal_energy_grad(x_input, num_elem, stiffness, elem_leng, boundary):
    """d/dx of _cal_energy; MATLAB gets this from `jacobian`."""
    x = np.asarray(x_input, dtype=np.float64).reshape((num_elem - 1, 2))
    rod_pos = np.concatenate([boundary[0:1, :], x, boundary[1:2, :]], axis=0)

    d = rod_pos[1:, :] - rod_pos[:-1, :]
    L = np.sqrt(np.sum(d ** 2, axis=1))
    # dE/d(segment vector) = stiffness * (L - elem_leng) * d / L
    coef = stiffness * (L - elem_leng) / np.where(L == 0, 1.0, L)
    seg = coef[:, None] * d

    # node k (interior, 0..num_elem-2) is the end of segment k and start of k+1
    return (seg[:-1, :] - seg[1:, :]).ravel()


def _cal_energy_len_grad(l, num_elem, stiffness, elem_leng):
    l = np.asarray(l, dtype=np.float64).ravel()
    return stiffness * (l - elem_leng)


def toy_rod_model():
    from scipy.optimize import minimize

    num_elem = 30
    stiffness = 300
    elem_leng = 40
    boundary = np.array([[-30.0, 0.0], [30.0, 0.0]])

    # --- position ---
    energy_fun = lambda x: _cal_energy(x, num_elem, stiffness, elem_leng, boundary)
    grad_fun = lambda x: _cal_energy_grad(x, num_elem, stiffness, elem_leng, boundary)

    # --- length ---
    energy_len_fun = lambda l: _cal_energy(l, num_elem, stiffness, elem_leng,
                                           boundary, True)
    grad_len_fun = lambda l: _cal_energy_len_grad(l, num_elem, stiffness, elem_leng)

    # --- multiple exprs ---
    expr_num = 10
    results_x_pos = np.zeros((num_elem - 1, 2 * expr_num))
    results_len = np.zeros((num_elem, expr_num))

    for expr in range(1, expr_num + 1):
        # --- init ---
        # rod_init = [25, 40]
        rod_init = (np.random.rand(num_elem - 1, 2) - 0.5) * 100
        # rod_init[:, 0] = np.arange(-(num_elem//2 - 1), num_elem//2)
        length_init = np.random.rand(num_elem) * 80

        # --- optimize ---
        rod_pos = np.concatenate([boundary[0:1, :], rod_init, boundary[1:2, :]], axis=0)

        res = minimize(lambda x: (energy_fun(x), grad_fun(x)), rod_init.ravel(),
                       jac=True, method='L-BFGS-B',
                       bounds=[(-100.0, 100.0)] * rod_init.size,
                       options={'disp': True})
        xopt = res.x.reshape(rod_init.shape)

        res_len = minimize(lambda l: (energy_len_fun(l), grad_len_fun(l)), length_init,
                           jac=True, method='L-BFGS-B',
                           bounds=[(0.0, 100.0)] * length_init.size,
                           options={'disp': True})
        xopt_len = res_len.x

        # --- save results ---
        results_x_pos[:, (expr - 1) * 2:expr * 2] = xopt
        results_len[:, expr - 1] = xopt_len

        # --- plot ---
        if num_elem == 2:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            X, Y = np.meshgrid(np.arange(-50, 51), np.arange(-50, 51))
            all_energy = np.zeros(X.shape)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    all_energy[i, j] = energy_fun([X[i, j], Y[i, j]])
            ax.plot_surface(X, Y, all_energy)

            grad_init = grad_fun(rod_init)
            end_grad = rod_init.ravel() + grad_init
            end_grad = end_grad / np.linalg.norm(end_grad)
            ax.plot([rod_init.ravel()[0], end_grad[0]],
                    [rod_init.ravel()[1], end_grad[1]],
                    [all_energy[int(round(rod_init.ravel()[1])) + 50,
                                int(round(rod_init.ravel()[0])) + 50],
                     all_energy[int(round(end_grad[1])) + 50,
                                int(round(end_grad[0])) + 50]], 'r-', linewidth=2)
            ax.scatter(round(xopt.ravel()[0]), round(xopt.ravel()[1]),
                       all_energy[int(round(xopt.ravel()[1])) + 50,
                                  int(round(xopt.ravel()[0])) + 50],
                       c='r', s=200)

    if False:
        import matplotlib.pyplot as plt
        plt.figure()
        rod_pos_final = np.concatenate([boundary[0:1, :], xopt, boundary[1:2, :]],
                                       axis=0)
        plt.plot(rod_pos_final[:, 0], rod_pos_final[:, 1], '-ok')

    return results_x_pos, results_len


if __name__ == '__main__':
    toy_rod_model()
