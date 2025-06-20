# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import torch
from typing import Annotated
from allo.ir.types import float32, int16, int32
import allo.dataflow as df
import numpy as np
from allo.memory import Layout
from allo.backend.experimental.external_kernel import ExternalModule

Ly = Layout("RR")

# Matrix dimensions
M = 64  # rows of A and C
K = 64  # cols of A, rows of B
N = 64  # cols of B and C


# PyTorch reference code starts
def matrix_multiply(
    A: Annotated[torch.Tensor, "shape: (64, 64)"],
    B: Annotated[torch.Tensor, "shape: (64, 64)"],
) -> Annotated[torch.Tensor, "shape: (64, 64)"]:
    """
    A: input matrix A (M x K)
    B: input matrix B (K x N)
    Returns: C = A @ B (M x N)
    """
    return torch.matmul(A, B)


# PyTorch reference code ends


def _test_matrix_multiply(kernel_path: str):

    kernel = ExternalModule(
        top="strange_kernel_int16",
        impl_path=kernel_path,
        input_idx=[0, 1],
        output_idx=[2],
    )

    Ty = int16

    @df.region()
    def top():
        @df.kernel(mapping=[1])
        def core(A: Ty[M, K] @ Ly, B: Ty[K, N] @ Ly, C: Ty[M, N] @ Ly):
            kernel(A, B, C)

    # Create random input matrices
    input_A = torch.randint(-10, 10, (M, K), dtype=torch.int16)
    input_B = torch.randint(-10, 10, (K, N), dtype=torch.int16)
    output = matrix_multiply(input_A, input_B).to(torch.int16)

    if "MLIR_AIE_INSTALL_DIR" in os.environ:
        mod = df.build(
            top,
            target="aie-mlir",
            profile=True,
            warmup=100,
            num_iters=1000,
        )
        output_allo = np.zeros((M, N)).astype(np.int16)
        mod(input_A.cpu().numpy(), input_B.cpu().numpy(), output_allo)

        # Gracefully handle verification
        try:
            np.testing.assert_allclose(output_allo, output, rtol=1e-2)
            print("PASS!")
        except AssertionError as e:
            print("FAIL!")
            print(f"Verification failed:\n{str(e)}")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


if __name__ == "__main__":
    _test_matrix_multiply("gemm.cc")
