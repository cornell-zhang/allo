# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import allo
from allo.ir.types import int32, float32


def test_math_scalar():
    M = 10
    K = 15
    N = 20
    A = np.float32(np.random.uniform(size=(M, K)))
    B = np.float32(np.random.uniform(size=(K, N)))

    def kernel(A: float32[M, K], B: float32[K, N]) -> float32[M, N]:
        C: float32[M, N] = 0.0
        D: float32[M, N] = 0.0
        for i, j in allo.grid(M, N):
            for k in allo.reduction(K):
                C[i, j] += A[i, k] * B[k, j]
        for i, j in allo.grid(M, N):
            D[i, j] = (allo.exp(C[i, j]) + allo.log(C[i, j])) / C[i, j]
        return D

    s = allo.customize(kernel)
    f = s.build()
    print(s.module)
    outs = np.zeros((M, N), dtype="float32")
    outs = f(A, B)
    np1 = np.matmul(A, B)
    np_outs = (np.exp(np1) + np.log(np1)) / np1
    np.testing.assert_allclose(outs, np_outs, atol=1e-3)


def test_linalg_matmul():
    M = 10
    K = 15
    N = 20
    np_0 = np.random.randint(0, 20, size=(M, K), dtype="int32")
    np_1 = np.random.randint(0, 20, size=(K, N), dtype="int32")

    # Test different ways to call matmul in order to make sure
    # the symbol resolver works correctly.
    from allo import matmul

    def kernel(A: int32[M, K], B: int32[K, N]) -> int32[M, N]:
        return matmul(A, B)

    s = allo.customize(kernel)
    f = s.build()
    np_out = kernel(np_0, np_1)
    allo_out = f(np_0, np_1)
    np.testing.assert_array_equal(allo_out, np_out)
    print(s.module)

    def kernel2(A: int32[M, K], B: int32[K, N]) -> int32[M, N]:
        return allo.matmul(A, B)

    s = allo.customize(kernel2)
    print(s.module)

    def kernel3(A: int32[M, K], B: int32[K, N]) -> int32[M, N]:
        return allo.matmul_error(A, B)

    with pytest.raises(AssertionError):
        s = allo.customize(kernel3)

    def kernel4(A: int32[M, K], B: int32[K, N]) -> int32[M, N]:
        return allo.dsl.matmul(A, B)

    s = allo.customize(kernel4)
    print(s.module)


def test_linalg_matmul_only2D():
    M = 10
    K = 15
    N = 20

    def kernel(A: int32[M, K, M, K], B: int32[M, K, K, N]) -> int32[M, K, M, N]:
        C = allo.matmul(A, B)
        return C

    with pytest.raises(AssertionError):
        allo.customize(kernel)


def test_linalg_matmul_nested():
    M = 10
    K = 15
    A = np.random.uniform(size=(M, K))
    B = np.random.uniform(size=(K, M))
    C = np.zeros((M, K), dtype="float32")

    def kernel() -> float32[M, K]:
        A1: float32[M, K] = A
        B1: float32[K, M] = B
        C1: float32[M, K] = C
        D = allo.matmul(allo.matmul(A1, B1), A1)
        # GeLU of matrix D
        for i, j in allo.grid(M, K):
            C1[i, j] = (
                0.5
                * D[i, j]
                * (
                    1.0
                    + allo.tanh(
                        allo.sqrt(2.0 / 3.1415926)
                        * (D[i, j] + 0.044715 * allo.power(D[i, j], 3.0))
                    )
                )
            )
        return C1

    s = allo.customize(kernel)
    print(s.module)
    f = s.build()
    outs = f()
    np_GeLU = kernel()
    np.testing.assert_allclose(outs, np_GeLU, atol=1e-4)


def test_linalg_batch_matmul():
    M = 10
    K = 20
    N = 25
    A = np.float32(np.random.uniform(size=(M, M, K)))
    B = np.float32(np.random.uniform(size=(M, K, N)))

    def kernel(A: float32[M, M, K], B: float32[M, K, N]) -> float32[M, M, N]:
        D = allo.bmm(A, B)
        return D

    s = allo.customize(kernel)
    print(s.module)
    f = s.build()

    outs = np.zeros((M, M, N), dtype="float32")
    outs = f(A, B)
    bmm_outs = kernel(A, B)
    np.testing.assert_allclose(outs, bmm_outs, atol=1e-4)


def test_linalg_batch_matmul_only3D():
    M = 10
    K = 20
    N = 25

    def kernel(A: float32[M, K, K, N], B: float32[M, K, N, M]) -> float32[M, K, K, M]:
        D = allo.bmm(A, B)
        return D

    with pytest.raises(AssertionError):
        allo.customize(kernel)


def test_linalg_batch_matmul_nested():
    M = 16
    K = 32
    N = 64
    A = np.random.randint(0, 20, size=(M, N, K), dtype="int32")
    B = np.random.randint(0, 20, size=(M, K, N), dtype="int32")

    def bmm(A: int32[M, N, K], B: int32[M, K, N]) -> int32[M, N, K]:
        C: int32[M, N, K]
        for i, j, k in allo.grid(M, N, K):
            C[i, j, k] = A[i, j, k] + 1
        D = allo.bmm(allo.bmm(A, B, name="bmm1"), C, name="bmm2")
        return D

    def top(A: int32[M, N, K], B: int32[M, K, N]) -> int32[M, N, K]:
        D = bmm(A, B)
        outputs = allo.bmm(allo.bmm(A, B, name="bmm3"), D, name="bmm4")
        return outputs

    s1 = allo.customize(bmm, lower_linalg=True)
    print(s1.module)

    loops = s1.get_loops()
    s1.split(loops.bmm1.L_0, 8)
    print(s1.module)

    loops = s1.get_loops()
    s1.reorder(loops.bmm1["L_0.outer"], loops.bmm1.L_1, loops.bmm1["L_0.inner"])
    s1.unroll(loops.bmm2.L_2)
    s1.fuse(loops.bmm1["L_0.outer"], loops.bmm1.L_1)
    print(s1.module)

    # Top-level
    s = allo.customize(top, lower_linalg=True)
    s.compose(s1)
    loops_ = s.get_loops()
    s.pipeline(loops_.bmm4.L_0)
    f = s.build()
    print(s.module)
    print(s.build("vhls"))

    outs = np.zeros((M, N, K), dtype="int32")
    outs = f(A, B)
    out_1 = np.einsum("ijk,ikn->ijn", A, B)
    out_2 = np.einsum("ijk,ikn->ijn", out_1, (A + 1))
    out_3 = np.einsum("ijk,ikn->ijn", out_1, out_2)
    np.testing.assert_allclose(outs, out_3, atol=1e-4)


def test_linalg_math():
    M = 10
    K = 15
    A = np.float32(np.random.uniform(size=(M, K)))
    B = np.float32(np.random.uniform(size=(K, M)))

    def kernel(A: float32[M, K], B: float32[K, M]) -> float32[M, M]:
        D = allo.matmul(A, B)
        C = (allo.add(allo.exp(D), allo.abs(D)) - allo.log(D)) / D
        return C

    s = allo.customize(kernel)
    f = s.build()
    print(s.module)
    outs = f(A, B)
    np_outs = kernel(A, B)
    np.testing.assert_allclose(outs, np_outs, atol=1e-3)


def test_linalg_softmax():
    # TODO: failed to lower to LLVM, see https://reviews.llvm.org/D153422
    M = 10
    K = 15

    def kernel(A: float32[M, K]) -> float32[M, K]:
        outs = allo.softmax(A)
        return outs

    with pytest.raises(AttributeError):
        allo.customize(kernel)


if __name__ == "__main__":
    pytest.main([__file__])
