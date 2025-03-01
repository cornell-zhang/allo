# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from allo.ir.types import float32
import allo.dataflow as df
import allo.backend.hls as hls
from allo.passes import generate_input_output_buffers
import numpy as np

M, N, K = 32, 32, 32
P0, P1 = 2, 2
Mt, Nt = M // P0, N // P1


@df.region()
def top():
    @df.kernel(mapping=[P0, P1])
    def gemm(A: float32[M, K], B: float32[K, N], C: float32[M, N]):
        pi, pj = df.get_pid()
        for i in range(pi * Mt, (pi + 1) * Mt):
            for j in range(pj * Nt, (pj + 1) * Nt):
                for k in range(K):
                    C[i, j] += A[i, k] * B[k, j]


def test_wrap_void():
    A = np.random.rand(M, K).astype(np.float32)
    B = np.random.rand(K, N).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)

    flatten = True
    s1 = df.customize(top)
    generate_input_output_buffers(s1.module, "top", flatten=flatten)
    module = str(s1.module)
    # Movement Function Generation
    assert (
        f"func.func @load_buf0(%arg0: memref<1024xf32>, %arg1: memref<32x32xf32>)"
        in module
    )
    assert (
        f"func.func @store_res2(%arg0: memref<32x32xf32>, %arg1: memref<1024xf32>)"
        in module
    )
    # Buffer Allocation
    assert f'%alloc = memref.alloc() {{name = "buf0"}} : memref<32x32xf32>' in module
    # Function Call
    assert (
        f"call @load_buf0(%arg0, %alloc) : (memref<1024xf32>, memref<32x32xf32>) -> ()"
        in module
    )
    assert (
        f"call @store_res2(%alloc_1, %arg2) : (memref<32x32xf32>, memref<1024xf32>) -> ()"
        in module
    )
    print("Data Movement (Flatten) Passed!")

    flatten = False
    s2 = df.customize(top)
    generate_input_output_buffers(s2.module, "top", flatten=flatten)
    module = str(s2.module)
    # Movement Function Generation
    assert (
        f"func.func @load_buf0(%arg0: memref<32x32xf32>, %arg1: memref<32x32xf32>)"
        in module
    )
    assert (
        f"func.func @store_res2(%arg0: memref<32x32xf32>, %arg1: memref<32x32xf32>)"
        in module
    )
    # Buffer Allocation
    assert f'%alloc = memref.alloc() {{name = "buf0"}} : memref<32x32xf32>' in module
    # Function Call
    assert (
        f"call @load_buf0(%arg0, %alloc) : (memref<32x32xf32>, memref<32x32xf32>) -> ()"
        in module
    )
    assert (
        f"call @store_res2(%alloc_1, %arg2) : (memref<32x32xf32>, memref<32x32xf32>) -> ()"
        in module
    )
    print("Data Movement (Non-flatten) Passed!")

    if hls.is_available("vitis_hls"):
        C = np.zeros((M, N), dtype=np.float32)
        mod = df.build(top)
        mod(A, B, C)
        np.testing.assert_allclose(C, np.dot(A, B), rtol=1e-5, atol=1e-5)
        print("Functionality Passed!")


def test_nowrap():
    A = np.random.rand(M, K).astype(np.float32)
    B = np.random.rand(K, N).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)

    s1 = df.customize(top)
    module = str(s1.module)
    assert "load_buf0" not in module
    assert "store_res2" not in module

    if hls.is_available("vitis_hls"):
        mod = df.build(top, target="vitis_hls", mode="csim", wrap_io=False)
        mod(A, B, C)
        np.testing.assert_allclose(C, np.dot(A, B), rtol=1e-5, atol=1e-5)
        print("Functionality Passed!")


if __name__ == "__main__":
    test_wrap_void()
    test_nowrap()
