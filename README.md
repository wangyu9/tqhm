# Variational quasi-harmonic maps for computing diffeomorphisms

PyTorch port of the MATLAB code accompanying the paper:

- **Variational Quasi-Harmonic Maps for Computing Diffeomorphisms**.
  Yu Wang, Minghao Guo, Justin Solomon.
  _ACM Trans. on Graph. 42(4)_. _ACM SIGGRAPH 2023 (Journal Track)_.
  [OpenAccessPaper](https://dl.acm.org/doi/pdf/10.1145/3592105)

Given a triangle mesh and a prescribed boundary curve, the solver computes a
deformed mesh (hopefully) without inverted triangles. You may have to adjust the
hyperparameters — especially the number of iterations, the learning rates (step
size), and the type of parameterization of the matrix `A(x)` — as described in
the paper. The demo does not use the optional post-processing joint-space Newton
step (Sec. 9.2.1); it is only needed for a few ill-conditioned examples (see
`oracle_joint_hessian.py`).

The original MATLAB version(https://github.com/wangyu9/qhm) depends on gptoolbox and SuiteSparse; the relevant gptoolbox
functions are vendored here, and the SuiteSparse solver is replaced by cuDSS wrapped in tdss.py
(the pipeline runs in float64 on CUDA).

# main.py

Command-line front end for the variational Beltrami solver
(`core_variational_beltrami`). Same job as `demo.py`, but the fields of the
`args` struct are exposed as command-line flags. Running with no arguments
reproduces `demo.main('david_o_A')`.

## Usage

```bash
python main.py [mesh] [options]
```

`mesh` names a folder under `test_cases/` containing an `input.obj`. A handful
of names (`WeberZorin14_fig19`, `cross`, `Lshape`, ...) resolve to
`test_cases/Simple/<mesh>`; everything else (`david_o_A`, `bunny_i_H`,
`lucy_o_G`, ...) resolves to `test_cases/Letters/<mesh>`. Use `--folder PATH`
to point at a mesh folder directly.

```bash
python main.py                    # default mesh, david_o_A
python main.py WeberZorin14_fig19 # smallest mesh, ~1s smoke test
```

## Common flags

- `--solver-adam-lr LR [LR ...]` — vadam learning rate; one value for a
  constant rate, or a list for a per-iteration schedule (its length sets the
  outer-iteration count).
- `--max-iter N` — number of outer iterations.
- `--verbose N` — `0` silent; `1` (default) prints the per-iteration flip line
  (`Iter 0009, flipps 0000`); `2` also prints the per-inner-iteration energy
  line.
- `--vis` — render the resulting map when done.

Run `python main.py --help` for the full list.
