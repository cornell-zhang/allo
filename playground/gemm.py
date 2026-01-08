# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# GEMM example with memory annotations for XLS backend testing with memory annotations.

import numpy as np
import allo
from allo.ir.types import int32
from allo.memory import Memory


def gemm_memory(M: int, N: int, K: int):
    """
    Matrix multiplication: C = A @ B

    Uses memory annotations to specify storage types for arrays.
    """

    def kernel(
        A: int32[M, K] @ Memory(resource="BRAM", storage_type="RAM_1P"),
        B: int32[K, N] @ Memory(resource="BRAM", storage_type="RAM_1P"),
        C: int32[M, N] @ Memory(resource="BRAM", storage_type="RAM_1P"),
    ):
        # C = A @ B (in-place modification, no return)
        for i, j in allo.grid(M, N, name="gemm"):
            C[i, j] = 0
            for k in allo.grid(K, name="k"):
                C[i, j] += A[i, k] * B[k, j]

    return kernel


def test_gemm_memory():
    """Test GEMM with memory annotations using XLS backend."""
    M, N, K = 4, 4, 4

    # Create the kernel
    kernel = gemm_memory(M, N, K)

    # Customize and build
    s = allo.customize(kernel)

    # Print the module to see the IR
    print("=" * 70)
    print("MLIR Module:")
    print("=" * 70)
    print(s.module)
    print()

    # Build for XLS backend with memory annotations
    print("=" * 70)
    print("Building for XLS backend with memory annotations...")
    print("=" * 70)

    mod = s.build(target="xls", project="gemm_memory.prj", use_memory=True)

    # Print generated C++ code
    print("\n" + "=" * 70)
    print("Generated C++ Code:")
    print("=" * 70)
    print(mod.final_cpp)
    print()

    # Print RAM rewrites textproto if available
    if hasattr(mod, "rewrites_textproto") and mod.rewrites_textproto:
        print("=" * 70)
        print("RAM Rewrites Textproto:")
        print("=" * 70)
        mod.print_textproto()
        print()

    # Test with numpy
    print("=" * 70)
    print("Testing with NumPy...")
    print("=" * 70)

    A = np.random.randint(-10, 10, size=(M, K), dtype=np.int32)
    B = np.random.randint(-10, 10, size=(K, N), dtype=np.int32)
    C_expected = A @ B

    print(f"A shape: {A.shape}")
    print(f"B shape: {B.shape}")
    print(f"Expected C shape: {C_expected.shape}")
    print(f"\nA:\n{A}")
    print(f"\nB:\n{B}")
    print(f"\nExpected C (A @ B):\n{C_expected}")

    print("\nPassed! GEMM with memory annotations test completed!")
    print("  Check gemm_memory.prj/ for generated files:")
    print("    - test_block.cpp: Generated C++ code")
    print("    - rewrites.textproto: RAM configuration")


if __name__ == "__main__":
    test_gemm_memory()
