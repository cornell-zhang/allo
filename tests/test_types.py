# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
import allo
from allo.ir.types import Int, UInt, Float, uint1, int32, float32, index
import allo.ir.types as T


def test_int32_float32():
    def kernel(a: int32) -> float32:
        b: float32 = float(int(float(1)))
        c: float32 = float(int(float(a)))
        return b + c

    s = allo.customize(kernel)
    print(s.module)
    mod = s.build()
    assert mod(1) == kernel(1)


def test_int32_float32_casting():
    def kernel(a: int32) -> float32:
        b: int32 = int(float(1))
        c: float32 = float(a)
        return b + c

    s = allo.customize(kernel)
    print(s.module)
    mod = s.build()
    assert mod(1) == kernel(1)


def test_load_type():
    def kernel(A: allo.ir.types.Float(32)[32, 32]) -> Int(32)[32, 32]:
        B: int32[32, 32] = 0
        v: T.Int(32) = 1
        for i, j in allo.grid(32, 32):
            B[i, j] = int(A[i, j]) + v
        return B

    s = allo.customize(kernel)
    print(s.module)
    mod = s.build()
    np_A = np.random.rand(32, 32).astype(np.float32)
    np_B = mod(np_A)
    np.testing.assert_allclose(np_B, np_A.astype(np.int32) + 1, rtol=1e-5, atol=1e-5)


def test_arbitrary_bitwidth_gemm():
    M, N, K = 4, 4, 4

    # This test is to make sure the whole flow works properly.
    def gemm(A: Int(5)[M, K], B: Int(5)[K, N], C: Int(14)[M, N]):
        # Use grid_for with name annotation
        for i, j, k in allo.grid(M, N, K, name="C"):
            C[i, j] += A[i, k] * B[k, j]

    # 1. Create customization
    s = allo.customize(gemm)
    print(s.module)

    # 3. Build and run
    mod = s.build()
    np_A = np.random.randint(-10, 10, size=(M, K)).astype(np.int32)
    np_B = np.random.randint(-10, 10, size=(K, N)).astype(np.int32)
    np_C = np.matmul(np_A, np_B)
    np_C_allo = np.zeros((M, N), dtype=np.int32)
    mod(np_A, np_B, np_C_allo)
    np.testing.assert_allclose(np_C, np_C_allo, rtol=1e-5)


def test_arbitrary_bitwidth_gemm_alloc_output():
    M, N, K = 4, 4, 4
    T_IN, T_OUT = Int(4), Int(16)

    def gemm(A: T_IN[M, K], B: T_IN[K, N]) -> T_OUT[M, N]:
        C: T_OUT[M, N] = 0
        for i, j, k in allo.grid(M, N, K, name="C"):
            C[i, j] += A[i, k] * B[k, j]
        return C

    M, N, K = 2, 2, 2
    for T_IN, T_OUT in [
        (Int(3), Int(7)),
        (Int(4), Int(8)),
        (Int(5), Int(9)),
        (Int(7), Int(16)),
        (Int(8), Int(16)),
    ]:
        s = allo.customize(gemm)
        mod = s.build()
        np_A = np.random.randint(-4, 4, size=(M, K)).astype(np.int32)
        np_B = np.random.randint(-4, 4, size=(K, N)).astype(np.int32)
        np_C = np.matmul(np_A, np_B)
        np_C_allo = mod(np_A, np_B)
        np.testing.assert_allclose(np_C, np_C_allo, rtol=1e-5)
        print(f"Passed {T_IN}, {T_OUT}!")

    # Note: UInt is still not supported
    M, N, K = 2, 2, 2
    for T_IN, T_OUT in [
        (UInt(3), UInt(7)),
        (UInt(4), UInt(8)),
        (UInt(5), UInt(9)),
        (UInt(7), UInt(16)),
        (UInt(8), UInt(16)),
    ]:
        s = allo.customize(gemm)
        mod = s.build()
        np_A = np.random.randint(0, 8, size=(M, K)).astype(np.int32)
        np_B = np.random.randint(0, 8, size=(K, N)).astype(np.int32)
        np_C = np.matmul(np_A, np_B)
        np_C_allo = mod(np_A, np_B)
        np.testing.assert_allclose(np_C, np_C_allo, rtol=1e-5)
        print(f"Passed {T_IN}, {T_OUT}!")

    M, N, K = 4, 4, 4
    for T_IN, T_OUT in [
        (Int(4), Int(15)),
        (Int(5), Int(16)),
        (Int(6), Int(17)),
        (Int(9), Int(31)),
        (Int(8), Int(32)),
        (Int(7), Int(33)),
        (Int(15), Int(34)),
        (Int(16), Int(32)),
        (Int(17), Int(31)),
    ]:
        s = allo.customize(gemm)
        mod = s.build()
        np_A = np.random.randint(-8, 8, size=(M, K)).astype(np.int32)
        np_B = np.random.randint(-8, 8, size=(K, N)).astype(np.int32)
        np_C = np.matmul(np_A, np_B)
        np_C_allo = mod(np_A, np_B)
        np.testing.assert_allclose(np_C, np_C_allo, rtol=1e-5)
        print(f"Passed {T_IN}, {T_OUT}!")


def test_load_type_scalar():
    def kernel(a: allo.ir.types.Float(32)) -> Int(32):
        b: Float(32) = float(int(1))
        c: float32 = int(a)
        return int(b + c)

    s = allo.customize(kernel)
    print(s.module)
    mod = s.build()
    assert mod(1.0) == kernel(1.0)


def test_bconv2D_nchw():
    bs = 4
    ic, oc = 6, 16
    ih, iw = 8, 8
    kh, kw = 3, 3
    oh, ow = ih - kh + 1, iw - kw + 1
    L = ic * kh * kw

    def bconv(
        A: uint1[bs, ic, ih, iw], F: uint1[oc, ic, kh, kw]
    ) -> int32[bs, oc, oh, ow]:
        B: int32[bs, oc, oh, ow] = 0
        for n, c, h, w in allo.grid(bs, oc, oh, ow):
            # popcount
            v: int32 = 0
            for rc, rh, rw in allo.reduction(ic, kh, kw):
                v += A[n, rc, h + rh, w + rw] ^ F[c, rc, rh, rw]
            B[n, c, h, w] = L - (v << 1)
        return B

    s = allo.customize(bconv)
    print(s.module)
    mod = s.build()
    np_A = np.random.randint(0, 2, size=(bs, ic, ih, iw))
    np_B = np.random.randint(0, 2, size=(oc, ic, kh, kw))
    np_C = np.zeros((bs, oc, oh, ow), np.int32)

    for n in range(0, bs):
        for c in range(0, oc):
            for y in range(0, oh):
                for x in range(0, ow):
                    for rc in range(0, ic):
                        for rh in range(0, kh):
                            for rw in range(0, kw):
                                np_C[n][c][y][x] += 1 - 2 * (
                                    np_A[n][rc][y + rh][x + rw] ^ np_B[c][rc][rh][rw]
                                )

    allo_C = mod(np_A, np_B)
    assert np.array_equal(np_C, allo_C)
    print("Passed!")


def test_avgpool_nchw():
    bs = 4
    ic, oc = 16, 16
    ih, iw = 8, 8
    kh, kw = 2, 2
    stride = 2
    oh, ow = (ih - kh) // stride + 1, (iw - kw) // stride + 1

    def avgpool_nchw(A: float32[bs, ic, ih, iw], B: float32[bs, oc, oh, ow]):
        stride: index = 2
        for n, c, h, w in allo.grid(bs, oc, oh, ow):
            v: float32 = 0.0
            for rh, rw in allo.reduction(kh, kw):
                v += A[n, c, h * stride + rh, w * stride + rw]
            B[n, c, h, w] = v / (kh * kw)

    s = allo.customize(avgpool_nchw)
    print(s.module)
    mod = s.build()
    np_A = np.random.rand(bs, ic, ih, iw).astype(np.float32)
    np_B = np.zeros((bs, oc, oh, ow), dtype=np.float32)
    mod(np_A, np_B)
    np_C = np.zeros((bs, oc, oh, ow), dtype=np.float32)
    avgpool_nchw(np_A, np_C)
    np.testing.assert_allclose(np_B, np_C, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__])
