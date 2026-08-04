"""reuse_struct.m -- mutable handle object caching everything that depends only
on the mesh connectivity (sparsity patterns, the tdss symbolic factorization,
the gradient/divergence operators)."""


class reuse_struct:
    def __init__(self):
        self.mode = 'none'   # or 'periodic'
        self.period = 1
        self.inited = 0
        self.Lc = None
        self.Qc = None
        self.count = 0

        self.RL = None
        self.q = None
        self.Q = None
        self.grad_xy = None
        self.div_xy = None

        self.fine_recorder = None

        # tdss-specific cached state (no MATLAB counterpart: replaces the
        # SuiteSparse `analyze` + `lchol` symbolic object).
        self.solver = None
        self.sub = None
