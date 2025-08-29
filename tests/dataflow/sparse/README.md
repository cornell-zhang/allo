<!--- Copyright Allo authors. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->
## Sparse Dense Matrix Multiplication

We provide a row-wise systolic array sparse dense matrix multiplication Allo implementations under the [`sparse`](sparse/) folder. For example, you can directly run the SEMM kernel using the CPU backend by typing the following command:
```bash
python3 test_sparse_systolic_data_packing.py
```

# Examples - Sparse Dense Matrix Multiplication

This folder contains examples of using Allo to design hardware accelerators for sparse dense matrix multiplication following a 2:4 sparsity pattern.

This hardware accelerator was designed to multiply sparse matrix A and dense matrix B to result in C.

`test_simple_sparse_systolic.py` implements a general matrix multiplication using a systolic array with a simple zero check before multiply accumulating values within A and B.

`test_sparse_systolic_data_packing.py` implements a row-wise product and cherry picks datapacked values in B that correspond to nonzero values in A, multiply accumulating them to result in C.

More info can be found in [this discussion thread](https://github.com/cornell-zhang/allo/discussions/289) and [this fork](https://github.com/CynyuS/allo-sparse).