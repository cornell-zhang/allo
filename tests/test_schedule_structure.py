# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import allo
from allo.ir.types import index, int32


@pytest.mark.parametrize("axes", [[0], [1], [0, 1]])
def test_matrix_add(axes):
    M, N = 2, 2

    def matrix_add(A: int32[M, N]) -> int32[M, N]:
        B: int32[M, N]
        for i, j in allo.grid(M, N, name="PE"):
            B[i, j] = A[i, j] + 1
        return B

    s = allo.customize(matrix_add)
    print(s.module)
    s.unfold("PE", axes=axes)
    assert str(s.module).count('from = "A"') == len(axes) * M
    assert str(s.module).count('to = "B"') == len(axes) * M
    print(s.module)
    np_A = np.random.randint(0, 10, size=(M, N))
    np_B = np_A + 1
    mod = s.build()
    res_allo = mod(np_A)
    np.testing.assert_allclose(res_allo, np_B)


@pytest.mark.parametrize("axes", [[0], [1], [0, 1]])
def test_matrix_add_function(axes):
    M, N = 2, 2

    def kernel(A: int32[M, N], B: int32[M, N], i: index, j: index):
        B[i, j] = A[i, j] + 1

    def matrix_add(A: int32[M, N]) -> int32[M, N]:
        B: int32[M, N]
        for i, j in allo.grid(M, N, name="PE"):
            kernel(A, B, i, j)
        return B

    s = allo.customize(matrix_add)
    print(s.module)
    s.unfold("PE", axes=axes)
    print(s.module)
    if axes == [0]:
        target_str = ["kernel"]
    elif axes == [1]:
        target_str = ["kernel_0", "kernel_1"]
    else:
        target_str = ["kernel_0_0", "kernel_0_1", "kernel_1_0", "kernel_1_1"]
    for t in target_str:
        assert f"call @{t}" in str(s.module)
    np_A = np.random.randint(0, 10, size=(M, N))
    np_B = np_A + 1
    mod = s.build()
    res_allo = mod(np_A)
    np.testing.assert_allclose(res_allo, np_B)


@pytest.mark.parametrize("axes", [[0], [1], [0, 1]])
def test_matmul(axes):
    M, N, K = 2, 2, 2

    def matmul(A: int32[M, K], B: int32[K, N]) -> int32[M, N]:
        C: int32[M, N] = 0
        for i, j in allo.grid(M, N, name="PE"):
            for k in range(K):
                C[i, j] += A[i, k] * B[k, j]
        return C

    s = allo.customize(matmul)
    print(s.module)
    s.unfold("PE", axes=axes)
    print(s.module)
    assert str(s.module).count('from = "A"') == len(axes) * M
    assert str(s.module).count('from = "B"') == len(axes) * M
    assert str(s.module).count('to = "C"') == len(axes) * M
    np_A = np.random.randint(0, 10, size=(M, K))
    np_B = np.random.randint(0, 10, size=(K, N))
    mod = s.build()
    res_allo = mod(np_A, np_B)
    np.testing.assert_allclose(res_allo, np_A @ np_B)


@pytest.mark.parametrize("axes", [[0], [1], [0, 1]])
def test_matmul_function(axes):
    M, N, K = 2, 2, 2

    def kernel(A: int32[M, K], B: int32[K, N], C: int32[M, N], i: index, j: index):
        for k in range(K):
            C[i, j] += A[i, k] * B[k, j]

    def matmul(A: int32[M, K], B: int32[K, N]) -> int32[M, N]:
        C: int32[M, N] = 0
        for i, j in allo.grid(M, N, name="PE"):
            kernel(A, B, C, i, j)
        return C

    s = allo.customize(matmul)
    print(s.module)
    s.unfold("PE", axes=axes)
    print(s.module)
    if axes == [0]:
        target_str = ["kernel"]
    elif axes == [1]:
        target_str = ["kernel_0", "kernel_1"]
    else:
        target_str = ["kernel_0_0", "kernel_0_1", "kernel_1_0", "kernel_1_1"]
    for t in target_str:
        assert f"call @{t}" in str(s.module)
    np_A = np.random.randint(0, 10, size=(M, K))
    np_B = np.random.randint(0, 10, size=(K, N))
    mod = s.build()
    res_allo = mod(np_A, np_B)
    np.testing.assert_allclose(res_allo, np_A @ np_B)


if __name__ == "__main__":
    pytest.main([__file__])
