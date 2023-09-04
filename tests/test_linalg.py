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


def test_linalg_matmul_5D():
    M = 10
    K = 15
    N = 20
    A = np.random.randint(0, 20, size=(N, M, K, M, K), dtype="int32")
    B = np.random.randint(0, 20, size=(N, M, K, K, N), dtype="int32")

    def kernel(
        A: int32[N, M, K, M, K], B: int32[N, M, K, K, N]
    ) -> int32[N, M, K, M, N]:
        C = allo.matmul(A, B)
        return C

    s = allo.customize(kernel)
    f = s.build()
    print(s.module)
    np_out = kernel(A, B)
    allo_out = f(A, B)
    np.testing.assert_array_equal(allo_out, np_out)


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


@pytest.mark.parametrize("enable_tensor", [True, False])
def test_linalg_batch_matmul(enable_tensor):
    M = 10
    K = 20
    N = 25
    A = np.float32(np.random.uniform(size=(M, M, K)))
    B = np.float32(np.random.uniform(size=(M, K, N)))

    def kernel(A: float32[M, M, K], B: float32[M, K, N]) -> float32[M, M, N]:
        D = allo.bmm(A, B)
        return D

    s = allo.customize(kernel, enable_tensor=enable_tensor)
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


@pytest.mark.parametrize("enable_tensor", [True, False])
def test_linalg_math(enable_tensor):
    M = 10
    K = 15
    A = np.float32(np.random.uniform(size=(M, K)))
    B = np.float32(np.random.uniform(size=(K, M)))

    def kernel(A: float32[M, K], B: float32[K, M]) -> float32[M, M]:
        D = allo.matmul(A, B)
        C = (allo.add(allo.exp(D), allo.abs(D)) - allo.log(D)) / D
        return C

    s = allo.customize(kernel, enable_tensor=enable_tensor)
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

    s = allo.customize(kernel)
    print(s.module)


def test_linalg_Linear_layer():
    inp_num = 12
    inp_len = 768
    Max_size = 12
    inp = np.float32(np.random.uniform(size=(inp_num, inp_len)))
    W = np.float32(np.random.uniform(size=(inp_len, inp_len)))
    B = np.float32(np.random.uniform(size=(inp_num, inp_len)))

    def Linear_layer(
        inp: float32[inp_num, inp_len],
        W: float32[inp_len, inp_len],
        B: float32[inp_num, inp_len],
    ) -> float32[inp_num, inp_len]:
        out = allo.add(allo.matmul(inp, W.T, name="gemm"), B, name="bias")
        return out

    s_q = allo.customize(Linear_layer, lower_linalg=True)
    print(s_q.module)
    loops = s_q.get_loops()
    s_q.pipeline(loops.bias.L_1)
    s_q.split(loops.gemm.L_1, Max_size)
    loops = s_q.get_loops()
    s_q.reorder(
        loops.gemm["L_1.outer"], loops.gemm.L_2, loops.gemm.L_0, loops.gemm["L_1.inner"]
    )
    s_q.pipeline(loops.gemm.L_2)
    print(s_q.module)
    print(s_q.build("vhls"))
    f = s_q.build()
    outs = f(inp, W, B)
    np_outs = Linear_layer(inp, W, B)
    np.testing.assert_allclose(outs, np_outs, atol=1e-3)


@pytest.mark.parametrize("enable_tensor", [True, False])
def test_linalg_transpose_3D(enable_tensor):
    M = 32
    K = 64
    N = 128
    A = np.random.randint(0, 20, size=(M, N, K), dtype="int32")
    B = np.random.randint(0, 20, size=(N, K, M), dtype="int32")

    def kernel(A: int32[M, N, K], B: int32[N, K, M]) -> int32[N, N, M]:
        C = allo.bmm(A, B.T).T
        return C

    s = allo.customize(kernel, enable_tensor=enable_tensor)
    f = s.build()
    np_out = kernel(A, B)
    allo_out = f(A, B)
    np.testing.assert_array_equal(allo_out, np_out)
    print(s.module)


@pytest.mark.parametrize("enable_tensor", [True, False])
def test_linalg_broadcast_scalar(enable_tensor):
    M = 10
    K = 15
    A = np.random.uniform(size=(M, K)).astype(np.float32)
    B = np.random.uniform(size=(K, M)).astype(np.float32)

    def kernel(A: float32[M, K], B: float32[K, M]) -> float32[M, M]:
        D = allo.matmul(A + 1, B)
        return D

    s = allo.customize(kernel, enable_tensor=enable_tensor)
    print(s.module)
    f = s.build()
    outs = f(A, B)
    np_outs = kernel(A, B)
    np.testing.assert_allclose(outs, np_outs, atol=1e-3)


@pytest.mark.parametrize("enable_tensor", [True, False])
def test_linalg_broadcast_tensor(enable_tensor):
    M, K, N = 10, 15, 12
    X = np.random.uniform(size=(M, K)).astype(np.float32)
    A = np.random.uniform(size=(N, K)).astype(np.float32)
    B = np.random.uniform(size=(N,)).astype(np.float32)

    def kernel(X: float32[M, K], A: float32[N, K], B: float32[N]) -> float32[M, N]:
        return (allo.matmul(X, A.T) + B) * 2

    s = allo.customize(kernel, enable_tensor=enable_tensor)
    print(s.module)
    f = s.build()
    outs = f(X, A, B)
    np_outs = kernel(X, A, B)
    np.testing.assert_allclose(outs, np_outs, atol=1e-3)


@pytest.mark.parametrize("enable_tensor", [True, False])
def test_linalg_transpose(enable_tensor):
    M, K, N = 10, 15, 12
    X = np.random.uniform(size=(M, K, N)).astype(np.float32)

    def kernel(X: float32[M, K, N]) -> float32[N, M, K]:
        return allo.transpose(X.T, (0, 2, 1))

    s = allo.customize(kernel, enable_tensor=enable_tensor)
    print(s.module)
    f = s.build()
    np.testing.assert_allclose(f(X), kernel(X), atol=1e-3)


@pytest.mark.parametrize("enable_tensor", [True, False])
def test_linear_library_call(enable_tensor):
    M, K, N = 10, 15, 12
    X = np.random.uniform(size=(M, K)).astype(np.float32)
    A = np.random.uniform(size=(N, K)).astype(np.float32)
    B = np.random.uniform(size=(N,)).astype(np.float32)

    def kernel(X: float32[M, K], A: float32[N, K], B: float32[N]) -> float32[M, N]:
        return allo.linear(X, A, B)

    s = allo.customize(kernel, enable_tensor=enable_tensor)
    print(s.module)
    f = s.build()
    outs = f(X, A, B)
    np_outs = kernel(X, A, B)
    np.testing.assert_allclose(outs, np_outs, atol=1e-3)


def test_linalg_conv2d_nchw():
    N = 1
    C = 1
    H = 7
    W = 7
    F = 1
    FH = 3
    FW = 3
    OH = H - FH + 1
    OW = W - FW + 1
    np_0 = np.random.randint(0, 2, size=(N, C, H, W), dtype="int32")
    np_1 = np.random.randint(0, 2, size=(F, C, FH, FW), dtype="int32")

    def kernel(A: int32[N, C, H, W], B: int32[F, C, FH, FW]) -> int32[N, F, OH, OW]:
        C = allo.conv2d(A, B)
        return C

    s = allo.customize(kernel)
    f = s.build()
    outs = kernel(np_0, np_1)
    np_outs = f(np_0, np_1)
    np.testing.assert_allclose(outs, np_outs, atol=1e-3)


def test_linalg_maxpool_nchw():
    N = 1
    C = 1
    H = 7
    W = 7
    F = 1
    FH = 3
    FW = 3
    OH = H - FH + 1
    OW = W - FW + 1
    np_0 = np.random.randint(0, 1000, size=(N, C, H, W), dtype="int32")
    np_1 = np.random.randint(0, 10, size=(FH, FW), dtype="int32")

    def kernel(A: int32[N, C, H, W], B: int32[FH, FW]) -> int32[N, C, OH, OW]:
        C = allo.maxpool(A, B)
        return C

    s = allo.customize(kernel)
    f = s.build()
    np_outs = kernel(np_0, np_1)
    outs = f(np_0, np_1)
    print(np_outs)
    print(outs)
    np.testing.assert_allclose(outs, np_outs, atol=1e-3)


def test_linalg_sumpool_nchw():
    N = 1
    C = 1
    H = 7
    W = 7
    F = 1
    FH = 3
    FW = 3
    OH = H - FH + 1
    OW = W - FW + 1
    np_0 = np.random.randint(0, 3, size=(N, C, H, W), dtype="int32")
    np_1 = np.random.randint(0, 10, size=(FH, FW), dtype="int32")

    def kernel(A: int32[N, C, H, W], B: int32[FH, FW]) -> int32[N, C, OH, OW]:
        C = allo.sumpool(A, B)
        return C

    s = allo.customize(kernel)
    f = s.build()
    outs = kernel(np_0, np_1)
    np_outs = f(np_0, np_1)
    np.testing.assert_allclose(outs, np_outs, atol=1e-3)


@pytest.mark.parametrize("enable_tensor", [True, False])
def test_copy_arg(enable_tensor):
    M, N = 2, 2

    def kernel(inp: float32[M, N]) -> float32[M, N]:
        A = allo.copy(inp)
        C = allo.copy(A)
        return C

    s = allo.customize(kernel, enable_tensor=enable_tensor)
    print(s.module)

    mod = s.build()
    inp = np.ones((M, N)).astype(np.float32)
    outp = mod(inp)
    np.testing.assert_allclose(inp, outp, rtol=1e-5)


@pytest.mark.parametrize("enable_tensor", [True, False])
def test_copy_const(enable_tensor):
    M, N = 2, 2

    def kernel() -> float32[M, N]:
        A: float32[M, N] = 1.0
        C = allo.copy(A)
        return C

    s = allo.customize(kernel, enable_tensor=enable_tensor)
    print(s.module)

    mod = s.build()
    inp = np.ones((M, N)).astype(np.float32)
    outp = mod()
    np.testing.assert_allclose(inp, outp, rtol=1e-5)


@pytest.mark.parametrize("enable_tensor", [True, False])
def test_library_higher_dimension_ops(enable_tensor):
    M, N, K, L = 5, 4, 3, 2
    A = np.random.uniform(size=(M, K, L)).astype(np.float32)
    B = np.random.uniform(size=(N, K)).astype(np.float32)
    C = np.random.uniform(size=(N,)).astype(np.float32)

    def kernel(
        A: float32[M, K, L], B: float32[N, K], C: float32[N]
    ) -> float32[M, L * N]:
        output1 = allo.transpose(A, (0, 2, 1))
        output2 = allo.linear(output1, B, C)
        output = allo.view(output2, (5, 8))
        return output

    s = allo.customize(kernel, enable_tensor=enable_tensor)
    print(s.module)
    mod = s.build()
    outp = mod(A, B, C)
    np_outp = kernel(A, B, C)
    np.testing.assert_allclose(outp, np_outp, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__])
