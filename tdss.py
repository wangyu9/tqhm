# https://gemini.google.com/app/12000cadc82e4e28

import glob
import os

import torch
import torch.nn as nn
from nvmath.sparse.advanced import DirectSolver, DirectSolverOptions
import numpy as np
import scipy.sparse as sp


def _find_cudss_mtlayer() -> str | None:
    """Return the path to libcudss_mtlayer_gomp.so.* if it can be found
    in any of the standard nvmath/nvidia install locations, else None."""
    import site
    search_roots = site.getsitepackages() + [site.getusersitepackages()]
    for root in search_roots:
        hits = glob.glob(
            os.path.join(root, "**", "libcudss_mtlayer_gomp.so*"),
            recursive=True,
        )
        if hits:
            return hits[0]
    return None


def _make_direct_solver_options() -> DirectSolverOptions:
    """Build DirectSolverOptions with the multithreading library auto-detected."""
    mtlib = _find_cudss_mtlayer()
    return DirectSolverOptions(multithreading_lib=mtlib)


class CuDSSFactorizeAndSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A_data, B, solver_A, solver_AT, row_ptr_bd, col_ind_bd, permutation, use_J, J1, J2, A_dummy_values, B_dummy, AT_dummy_values, BT_dummy):
        
        # --- THE FIX: IN-PLACE MEMORY UPDATES ON DIRECT BUFFERS ---
        A_dummy_values.copy_(A_data.detach().reshape(-1))
        B_dummy.copy_(B.detach().reshape(B_dummy.shape))

        solver_A.factorize()
        X_flat = solver_A.solve() 
        
        X = X_flat.reshape(B.shape)

        ctx.save_for_backward(A_data, B, X, row_ptr_bd, col_ind_bd, permutation)
        ctx.solver_AT = solver_AT
        ctx.use_J = use_J
        ctx.J1 = J1
        ctx.J2 = J2
        ctx.AT_dummy_values = AT_dummy_values
        ctx.BT_dummy = BT_dummy
        
        return X

    @staticmethod
    def backward(ctx, grad_X):
        A_data, B, X, row_ptr_bd, col_ind_bd, permutation = ctx.saved_tensors
        solver_AT = ctx.solver_AT
        AT_dummy_values = ctx.AT_dummy_values
        BT_dummy = ctx.BT_dummy

        from cuda.core import Device
        device_idx = A_data.device.index if A_data.device.index is not None else 0
        Device(device_idx).set_current()

        A_data_transposed = A_data.detach().reshape(-1)[permutation]

        # In-place copy for the backward solver
        AT_dummy_values.copy_(A_data_transposed)
        BT_dummy.copy_(grad_X.contiguous().reshape(BT_dummy.shape))

        solver_AT.factorize()
        V_flat = solver_AT.solve()
        
        # ==========================================
        # GRADIENT W.R.T A_data
        # ==========================================
        if ctx.use_J:
            # --- THE FIX: STRICT 2D SHAPE ENFORCEMENT ---
            V_mat = V_flat.reshape(BT_dummy.shape)
            X_mat = X.reshape(BT_dummy.shape)
            
            # Force them to be 2D column vectors if they are 1D
            if V_mat.dim() == 1:
                V_mat = V_mat.unsqueeze(1)
                X_mat = X_mat.unsqueeze(1)
            # --------------------------------------------
                
            J1_V = ctx.J1 @ V_mat
            J2_X = ctx.J2 @ X_mat
            
            # Now (12, 1) * (12, 1) safely yields (12, 1) without broadcasting
            grad_A_matrix = -(J2_X * J1_V)
            
            grad_A_data_flat = grad_A_matrix.squeeze(1) if BT_dummy.dim() == 1 else grad_A_matrix.sum(dim=-1)
            
        else:
            num_rows = row_ptr_bd.size(0) - 1
            elements_per_row = row_ptr_bd[1:] - row_ptr_bd[:-1]
            i = torch.repeat_interleave(torch.arange(num_rows, device=A_data.device), elements_per_row)
            j = col_ind_bd

            X_flat_reshaped = X.reshape(BT_dummy.shape)
            
            if V_flat.dim() == 1:
                grad_A_data_flat = -V_flat[i] * X_flat_reshaped[j]
            else:
                grad_A_data_flat = -(V_flat[i] * X_flat_reshaped[j]).sum(dim=-1)

        grad_A_data = grad_A_data_flat.reshape(A_data.shape)

        # 14 arguments in forward() -> 14 return values in backward()
        return grad_A_data, V_flat.reshape(B.shape), None, None, None, None, None, None, None, None, None, None, None, None


class BatchedAsymmetricSparseSolverLayer(nn.Module):

    def __init__(self, row_ptr, col_ind, num_rows, batch_size, rhs_columns=1, use_J_formulation=True):
        super().__init__()
        
        self.batch_size = batch_size
        self.num_rows = num_rows
        self.use_J_formulation = use_J_formulation
        nnz = col_ind.shape[0]

        counts = row_ptr[1:] - row_ptr[:-1]
        counts_bd = counts.repeat(batch_size)
        total_rows = batch_size * num_rows
        
        row_ptr_bd = torch.zeros(total_rows + 1, dtype=row_ptr.dtype, device=row_ptr.device)
        row_ptr_bd[1:] = torch.cumsum(counts_bd, dim=0)

        col_ind_bd = col_ind.repeat(batch_size)
        offsets = (torch.arange(batch_size, device=col_ind.device) * num_rows).repeat_interleave(nnz)
        col_ind_bd += offsets

        self.register_buffer('row_ptr_bd', row_ptr_bd)
        self.register_buffer('col_ind_bd', col_ind_bd)

        total_nnz = batch_size * nnz

        if use_J_formulation:
            row_indices_bd = torch.repeat_interleave(torch.arange(total_rows, device=row_ptr.device), counts_bd)
            nnz_indices = torch.arange(total_nnz, device=row_ptr.device)
            ones = torch.ones(total_nnz, dtype=torch.float64, device=row_ptr.device)
            
            J1_coo_indices = torch.stack([nnz_indices, row_indices_bd])
            self.J1 = torch.sparse_coo_tensor(J1_coo_indices, ones, size=(total_nnz, total_rows)).to_sparse_csr()
            
            J2_coo_indices = torch.stack([nnz_indices, col_ind_bd])
            self.J2 = torch.sparse_coo_tensor(J2_coo_indices, ones, size=(total_nnz, total_rows)).to_sparse_csr()
        else:
            self.J1 = None
            self.J2 = None

        idx_data = np.arange(total_nnz, dtype=np.int32)
        import scipy.sparse as sp
        A_idx_csr = sp.csr_matrix(
            (idx_data, col_ind_bd.cpu().numpy(), row_ptr_bd.cpu().numpy()),
            shape=(total_rows, total_rows)
        )
        AT_idx_csr = A_idx_csr.transpose().tocsr()
        
        self.register_buffer('transpose_permutation', torch.from_numpy(AT_idx_csr.data).long())
        row_ptr_AT = torch.from_numpy(AT_idx_csr.indptr).to(row_ptr.device)
        col_ind_AT = torch.from_numpy(AT_idx_csr.indices).to(row_ptr.device)
        
        # --- THE FIX: KEEP DIRECT REFERENCES TO THE MEMORY BUFFERS ---
        b_shape = (total_rows, rhs_columns) if rhs_columns > 1 else (total_rows,)
        # b_shape = (total_rows, rhs_columns) # wangyu: always keep the size-1 dim
        
        self.register_buffer('A_dummy_values', torch.ones(total_nnz, dtype=torch.float64, device=row_ptr.device))
        self.register_buffer('B_dummy', torch.ones(b_shape, dtype=torch.float64, device=row_ptr.device))
        
        self.register_buffer('AT_dummy_values', torch.ones(total_nnz, dtype=torch.float64, device=row_ptr.device))
        self.register_buffer('BT_dummy', torch.ones(b_shape, dtype=torch.float64, device=row_ptr.device))
        # -------------------------------------------------------------

        A_csr = torch.sparse_csr_tensor(
            crow_indices=self.row_ptr_bd,
            col_indices=self.col_ind_bd,
            values=self.A_dummy_values,
            size=(total_rows, total_rows)
        )

        if False:
          
            # wangyu. remember PyTorch tensors, by default, are row-major. 
            b_tmp = self.B_dummy
            print(f'BatchedAsymmetricSparseSolverLayer: A_csr:{A_csr.shape}, B:{b_tmp.shape}')
            solver_A_tmp = DirectSolver(a=A_csr, b=b_tmp)

            '''
            # The correct way to get a (N, 1) column-major tensor from a 1D tensor:
            # 1. View it as (1, N) -> row-major with strides (N, 1)
            # 2. Transpose it to (N, 1) -> column-major with strides (1, N)
            '''
            b_tmp = self.B_dummy[...,None]
            # b_tmp = b_tmp.transpose(-2, -1).contiguous().transpose(-2, -1) 
            b_tmp = b_tmp.view(1, -1).transpose(0, 1)
            print(f'BatchedAsymmetricSparseSolverLayer: A_csr:{A_csr.shape}, B:{b_tmp.shape}')
            solver_A_tmp = DirectSolver(a=A_csr, b=b_tmp)

            b_tmp = torch.stack([self.B_dummy,self.B_dummy], dim=-1)
            # Convert to column-major memory layout as required by DirectSolver
            b_tmp = b_tmp.transpose(-2, -1).contiguous().transpose(-2, -1)
            print(f'BatchedAsymmetricSparseSolverLayer: A_csr:{A_csr.shape}, B:{b_tmp.shape}')
            solver_A_tmp = DirectSolver(a=A_csr, b=b_tmp)


        opts = _make_direct_solver_options()
        self.solver_A = DirectSolver(a=A_csr, b=self.B_dummy, options=opts)
        self.solver_A.plan()

        AT_csr = torch.sparse_csr_tensor(
            crow_indices=row_ptr_AT,
            col_indices=col_ind_AT,
            values=self.AT_dummy_values,
            size=(total_rows, total_rows)
        )
        self.solver_AT = DirectSolver(a=AT_csr, b=self.BT_dummy, options=opts)
        self.solver_AT.plan()

    def forward(self, A_data, B):

        # print(self, "input shapes: ", A_data.shape, B.shape)
        return CuDSSFactorizeAndSolve.apply(
            A_data,
            B,
            self.solver_A,
            self.solver_AT,
            self.row_ptr_bd,
            self.col_ind_bd,
            self.transpose_permutation,
            self.use_J_formulation,
            self.J1,
            self.J2,
            self.A_dummy_values,  # Pass the explicit buffers!
            self.B_dummy,
            self.AT_dummy_values,
            self.BT_dummy
        )


class BatchedAsymmetricSparseSolverLayer2(nn.Module):
    def __init__(self, row_ptr, col_ind, num_rows, batch_size, rhs_columns=1, use_J_formulation=True):
        super().__init__()

        self.num_groups = batch_size
        self.group_size = 1

        self.layers = nn.ModuleList([
            BatchedAsymmetricSparseSolverLayer(
                row_ptr=row_ptr, 
                col_ind=col_ind, 
                num_rows=num_rows, 
                batch_size=self.group_size,
                rhs_columns=rhs_columns,
                use_J_formulation=use_J_formulation 
            ) for _ in range(self.num_groups)
        ])

        # --- THE FIX: Create a dedicated GPU Stream for each item in the batch ---
        if torch.cuda.is_available():
            self.streams = [torch.cuda.Stream() for _ in range(self.num_groups)]
        else:
            self.streams = None

    def forward(self, A_data, B):
        assert len(A_data.shape) == 2 # [batch_size, nnz]
        assert len(B.shape) == 3 # [batch_size, n, rhs_columns]

        X_list = [None] * self.num_groups

        if self.streams is not None:
            # 1. Launch all solvers in parallel on independent streams
            for i in range(self.num_groups):
                with torch.cuda.stream(self.streams[i]):
                    X_list[i] = self.layers[i](
                        A_data[i:i+self.group_size, ...], 
                        B[i:i+self.group_size, ...]
                    )

            # 2. CRITICAL: Re-sync back to the main PyTorch stream.
            # We must force the default stream to wait for all our custom streams to 
            # finish before we try to concatenate the results, otherwise we will get garbage memory.
            current_stream = torch.cuda.current_stream()
            for s in self.streams:
                current_stream.wait_stream(s)
        else:
            # CPU Fallback (Sequential)
            for i in range(self.num_groups):
                X_list[i] = self.layers[i](
                    A_data[i:i+self.group_size, ...], 
                    B[i:i+self.group_size, ...]
                )

        # Concatenate the results safely on the main stream
        X = torch.cat(X_list, dim=0)

        return X


# ==========================================
# 3. TEST SCRIPT
# ==========================================


def compare_with_scipy(A_data, B, X_torch, row_ptr, col_ind, num_rows, batch_size, rhs_columns=1, tol=1e-5):
    """
    Compares the PyTorch sparse solver output against SciPy's sparse solver.
    """
      
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.linalg as splinalg

    print("\n--- Running SciPy Ground Truth Comparison ---")
    
    # Move tensors to CPU and convert to NumPy
    A_data_np = A_data.detach().cpu().numpy()
    B_np = B.detach().cpu().numpy()
    row_ptr_np = row_ptr.cpu().numpy()
    col_ind_np = col_ind.cpu().numpy()
    X_torch_np = X_torch.detach().cpu().numpy()

    X_scipy_list = []
    
    # Solve each matrix in the batch individually using SciPy
    for k in range(batch_size):
        # Reconstruct the k-th sparse matrix
        A_sp = sp.csr_matrix(
            (A_data_np[k], col_ind_np, row_ptr_np), 
            shape=(num_rows, num_rows)
        )
        
        # Solve A_k * X_k = B_k
        X_k = splinalg.spsolve(A_sp, B_np[k])
        
        # spsolve drops the column dimension if RHS is a 1D vector, so we reshape it back
        if rhs_columns == 1: # and B_np[k].ndim == 1:
            X_k = X_k[...,None] # wangyu
        
        # print(f'X_k: {X_k.shape}')
        X_scipy_list.append(X_k)

    # Stack the SciPy results to match our PyTorch batch shape
    X_scipy_np = np.stack(X_scipy_list)

    # print('shapes (torch,scipy):', X_torch_np.shape, X_scipy_np.shape)
    assert X_torch_np.shape == X_scipy_np.shape

    print("PyTorch X Values:\n", X_torch_np)
    print("\nSciPy X Values:\n", X_scipy_np)
    
    # Calculate the maximum absolute error
    max_diff = np.max(np.abs(X_scipy_np - X_torch_np))
    print(f"\nMax Absolute Difference between PyTorch and SciPy: {max_diff:.4e}")
    
    if max_diff < tol:
        print("✅ The PyTorch forward pass matches SciPy!")
        return True
    else:
        print("❌ Mismatch detected in the forward pass.")
        return False


def scipy_to_torch_csr(matrix, copy: bool = False, device=None) -> torch.Tensor:
    import scipy.sparse
    
    # https://gemini.google.com/app/d7a60123d54f4133
    """
    Converts a SciPy sparse matrix to a PyTorch sparse CSR tensor.

    Args:
        matrix: A SciPy sparse matrix (COO, CSR, CSC, LIL, etc.).
        copy (bool): If True, forces a copy of the underlying data. 
                     If False (default), shares memory with the NumPy arrays where possible.
        device: The desired device of returned tensor (e.g., 'cpu', 'cuda').

    Returns:
        torch.Tensor: A PyTorch sparse CSR tensor.
    """
    # 1. Ensure the matrix is in CSR format
    csr_mat = matrix.tocsr()
    
    # 2. Extract arrays and convert to PyTorch tensors
    if copy:
        # torch.tensor() forces a copy of the data
        crow_indices = torch.tensor(csr_mat.indptr, dtype=torch.int64)
        col_indices = torch.tensor(csr_mat.indices, dtype=torch.int64)
        values = torch.tensor(csr_mat.data)
    else:
        # torch.from_numpy() shares memory with the original SciPy matrix
        # Note: PyTorch expects int32 or int64 for indices. SciPy defaults 
        # to int32, which works perfectly with torch.from_numpy().
        crow_indices = torch.from_numpy(csr_mat.indptr)
        col_indices = torch.from_numpy(csr_mat.indices)
        values = torch.from_numpy(csr_mat.data)

    # 3. Construct and return the PyTorch sparse tensor
    return torch.sparse_csr_tensor(
        crow_indices=crow_indices,
        col_indices=col_indices,
        values=values,
        size=csr_mat.shape,
        device=device
    )

    
def test(A_sparse=None, B_dense=None, use_J_formulation=False, 
      check_grad=True, timing_runs=0, group_size=None):
    
    '''
    input: either both A, B are matrices
          X, _ = tdss.test(A, B, check_grad=False) 
      or A is a list of sparse matrices, B is a 3-dim tensor.
          X, _ = tdss.test([A], B[None,...], check_grad=False)
    '''

    if True:
      torch.set_default_dtype(torch.float64)

    # # Generate random data for A and B. 
    # # Add an offset (+1.0) so matrix A doesn't become singular during random generation
    # A_data = torch.rand(BATCH_SIZE, NNZ, dtype=torch.float64, device=DEVICE) + 1.0 

    # if RHS_COLUMNS > 1:
    #     B = torch.rand(BATCH_SIZE, NUM_ROWS, RHS_COLUMNS, dtype=torch.float64, device=DEVICE)
    # else:
    #     B = torch.rand(BATCH_SIZE, NUM_ROWS, dtype=torch.float64, device=DEVICE)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if A_sparse is not None and B_dense is not None:
        print("\n" + "="*50)
        print("TESTING WITH CUSTOM SPARSE TENSORS")
        print("="*50)
        
        # Extract topology and batch info from provided tensors
        if isinstance(A_sparse, list):
            BATCH_SIZE = len(A_sparse) # A_sparse.size(0)
            NUM_ROWS = A_sparse[0].size(0)
            row_ptr = A_sparse[0].crow_indices()
            col_ind = A_sparse[0].col_indices()
            A_data = torch.stack([A_sparse[i].values().detach().clone() for i in range(len(A_sparse))])
            assert B_dense.dim() == 3
            assert A_sparse[0].shape[0] == B_dense[0].shape[0]
            assert A_sparse[0].shape[0] == A_sparse[0].shape[1]
        elif A_sparse.dim() == 2:
            BATCH_SIZE = 1 
            assert B_dense.dim() <= 2
            NUM_ROWS = A_sparse.size(0)
            row_ptr = A_sparse.crow_indices()
            col_ind = A_sparse.col_indices()
            A_data = A_sparse.values().unsqueeze(0).expand(BATCH_SIZE, -1).detach().clone()
        else:
            raise ValueError("A_sparse must be a 2D or 3D CSR tensor.")

        if B_dense.dim() == 3:
            RHS_COLUMNS = B_dense.size(2)
        elif B_dense.dim() == 2 and BATCH_SIZE == 1:
            RHS_COLUMNS = B_dense.size(1)
            B_dense = B_dense.unsqueeze(0)
        else:
            RHS_COLUMNS = 1

        B = B_dense.detach().clone()
        A_data.requires_grad_(True)
        B.requires_grad_(True)
        NNZ = len(col_ind)
        
    else:

        print("\n" + "="*50)
        print("TESTING WITH DEFAULT HARDCODED MATRICES")
        print("="*50)
        
        # Fallback to defaults
        NUM_ROWS = 3
        BATCH_SIZE = 2
        RHS_COLUMNS = 1

        # Define a simple asymmetric 3x3 sparsity pattern (CSR format)
        # [ X  X  0 ]
        # [ 0  X  X ]
        # [ X  0  X ]
        row_ptr = torch.tensor([0, 2, 4, 6], dtype=torch.int32, device=DEVICE)
        col_ind = torch.tensor([0, 1, 1, 2, 0, 2], dtype=torch.int32, device=DEVICE)
        NNZ = len(col_ind)

        # Batch 0: [4, 1], [5, 2], [1, 6]
        # Batch 1: [5, 2], [6, 3], [2, 7]  (Slightly different, but still diagonally dominant)
        A_values = [
            [4.0, 1.0, 5.0, 2.0, 1.0, 6.0],
            [5.0, 2.0, 6.0, 3.0, 2.0, 7.0]
        ]
        A_data = torch.tensor(A_values, dtype=torch.float64, device=DEVICE)
        A_data.requires_grad_(True)

        B_values = [
            [[1.0], [2.0], [3.0]],
            [[4.0], [5.0], [6.0]]
        ]
        # if RHS_COLUMNS > 1:
        #     B = torch.tensor(B_values, dtype=torch.float64, device=DEVICE).unsqueeze(-1).repeat(1, 1, RHS_COLUMNS)
        # else:
        #     B = torch.tensor(B_values, dtype=torch.float64, device=DEVICE)

        B = torch.tensor(B_values, dtype=torch.float64, device=DEVICE)
        B.requires_grad_(True)

    # print(f"test():shapes: {A_data.shape, B.shape}")

    assert len(A_data.shape)==2 # [batch_size, nnz]
    assert len(B.shape)==3 # [batch_size, n, rhs_columns]

    print("Initializing CuPy-Free batched solver layer...")
    if group_size==None:
        layer = BatchedAsymmetricSparseSolverLayer(
            row_ptr=row_ptr, 
            col_ind=col_ind, 
            num_rows=NUM_ROWS, 
            batch_size=BATCH_SIZE,
            rhs_columns=RHS_COLUMNS,
            use_J_formulation=use_J_formulation # Set to False for faster production use
        ).to(DEVICE)
    else:
        layer = BatchedAsymmetricSparseSolverLayer2(
            row_ptr=row_ptr, 
            col_ind=col_ind, 
            num_rows=NUM_ROWS, 
            batch_size=BATCH_SIZE,
            rhs_columns=RHS_COLUMNS,
            use_J_formulation=use_J_formulation # Set to False for faster production use
        ).to(DEVICE)

    # 1. Test Forward Pass
    print("\nRunning Forward Pass...")
    X = layer(A_data, B)
    print("Forward Pass successful. Output Shape:", X.shape)

    if True:
        # ==========================================
        # 1. Test Forward Pass & Print X
        # ==========================================
        print("\n--- Running Forward Pass ---")
        X = layer(A_data, B)
        print("X Output Shape:", X.shape)
        print("X Values:\n", X)

        # ==========================================
        # 2. Test Manual Backward Pass & Print Gradients
        # ==========================================
        print("\n--- Running Manual Backward Pass ---")
        # Create a scalar loss so we can trigger backpropagation
        loss = X.sum() 
        loss.backward()

        print("Gradient w.r.t A_data Shape:", A_data.grad.shape)
        print("Gradient w.r.t A_data:\n", A_data.grad)

        # CRITICAL: We must clear the accumulated gradients before running gradcheck
        A_data.grad = None
        B.grad = None

    if True:
        passed_scipy = compare_with_scipy(
              A_data=A_data, 
              B=B, 
              X_torch=X, 
              row_ptr=row_ptr, 
              col_ind=col_ind, 
              num_rows=NUM_ROWS, 
              batch_size=BATCH_SIZE, 
              rhs_columns=RHS_COLUMNS
            )

    if check_grad:

        # 2. Test Backward Pass (Gradcheck)
        print("\nRunning PyTorch Gradcheck (Numerical vs Analytical gradient comparison)...")
        print("This may take a few seconds...")
        
        # We use slightly relaxed tolerances due to cuSPARSE atomic additions in the J1/J2 formulation
        test_passed = torch.autograd.gradcheck(
            layer, 
            (A_data, B), 
            eps=1e-6, 
            atol=1e-6, 
            nondet_tol=1e-7
        )
        
        if test_passed:
            print("✅ Gradcheck PASSED! Your custom backward pass is mathematically correct.")
        else:
            print("❌ Gradcheck FAILED.")


    if timing_runs>0:
        # ==========================================
        # PERFORMANCE BENCHMARKING
        # ==========================================
        print("\n" + "="*50)
        print("PERFORMANCE BENCHMARKING (GPU Synchronized)")
        print("="*50)
        
        import time
        NUM_RUNS = timing_runs

        # 1. Warm-up (get CUDA contexts and memory allocations fully initialized)
        for _ in range(5):
            X_warmup = layer(A_data, B)
            X_warmup.sum().backward()
            A_data.grad = None
            B.grad = None

        fwd_times = []
        bwd_times = []

        # 2. Timing Loop
        for _ in range(NUM_RUNS):
            # --- Time Forward Pass ---
            if DEVICE.type == 'cuda': torch.cuda.synchronize()
            t0 = time.perf_counter()
            
            X_bench = layer(A_data, B)
            
            if DEVICE.type == 'cuda': torch.cuda.synchronize()
            t1 = time.perf_counter()
            fwd_times.append((t1 - t0) * 1000) # Convert to milliseconds

            loss_bench = X_bench.sum()

            # --- Time Backward Pass ---
            if DEVICE.type == 'cuda': torch.cuda.synchronize()
            t2 = time.perf_counter()
            
            loss_bench.backward()
            
            if DEVICE.type == 'cuda': torch.cuda.synchronize()
            t3 = time.perf_counter()
            bwd_times.append((t3 - t2) * 1000)

            # Clear gradients for the next loop
            A_data.grad = None
            B.grad = None

        avg_fwd = sum(fwd_times) / NUM_RUNS
        avg_bwd = sum(bwd_times) / NUM_RUNS

        print(f"Average Forward Pass:  {avg_fwd:.4f} ms")
        print(f"Average Backward Pass: {avg_bwd:.4f} ms")
        print(f"Total Step Time:       {avg_fwd + avg_bwd:.4f} ms over {NUM_RUNS} runs.")

    return X, None


def test1(A_scipy, B_scipy, bs=None):

    single_solve = (bs==None)

    A = scipy_to_torch_csr(A_scipy)
    B = torch.tensor(B_scipy)[:,0:1]

    print(B.shape, A.shape, A.nonzero)

    # bs=1 0.6320 ms bs =256 7.5610 ms  bs=1024 23.5407 ms

    if not single_solve:
        X, _ = test([A for i in range(bs)], B[None,...].repeat([bs, 1, 1]), check_grad=False, timing_runs=100)
    else:
        X, _ = test(A, B, check_grad=False, timing_runs=100)

    return X, _

if __name__ == "__main__":
    test()
