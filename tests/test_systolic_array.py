# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import allo
from allo.ir.types import int4, int8, int16, int32, float32, index, Int, UInt


def test_subview_systolic():
    M, N, K = 2, 2, 2

    def kernel(
        A_in: int8[K],
        B_in: int8[K],
        A_out: int8[K],
        B_out: int8[K],
        C: int16[M, N],
        i: index,
        j: index,
    ):
        for k in range(K):
            a: int8 = A_in[k]
            b: int8 = B_in[k]
            C[i, j] += a * b
            A_out[k] = a
            B_out[k] = b

    def systolic_array(A: int8[M, K], B: int8[K, N], C: int16[M, N]):
        A_fifo: int8[M, N + 1, K]
        B_fifo: int8[N, M + 1, K]

        for k in range(K, name="data_load"):
            for m in range(M):
                A_fifo[m, 0, k] = A[m, k]
            for n in range(N):
                B_fifo[n, 0, k] = B[k, n]
        for i, j in allo.grid(M, N, name="PE"):
            kernel(
                A_fifo[i, j], B_fifo[j, i], A_fifo[i, j + 1], B_fifo[j, i + 1], C, i, j
            )
        A_drain: int8[M]
        B_drain: int8[N]
        for k in range(K, name="data_drain"):
            for m in range(M):
                A_drain[m] = A_fifo[m, N, k]
            for n in range(N):
                B_drain[n] = B_fifo[n, M, k]

    s = allo.customize(systolic_array)
    print(s.module)

    mod = s.build()
    A = np.random.randint(-8, 8, size=(M, K)).astype(np.int8)
    B = np.random.randint(-8, 8, size=(K, N)).astype(np.int8)
    allo_C = np.zeros((M, N), dtype=np.int16)
    mod(A, B, allo_C)
    np_C = A.astype(np.int16) @ B.astype(np.int16)
    np.testing.assert_allclose(allo_C, np_C, atol=1e-3)


def test_subview_systolic_dsp_packed_int4xint4():
    M, N, K = 2, 2, 2

    def kernel(
        A_in: int4[K],
        B_in: int4[K],
        A_out: int4[K],
        B_out: int4[K],
        C: int8[M, N],
        i: index,
        j: index,
    ):
        for k in range(0, K, 2):
            a0: int4 = A_in[k]
            a1: int4 = A_in[k + 1]
            b0: int4 = B_in[k]
            b1: int4 = B_in[k + 1]
            a0u: UInt(4) = 0
            a1u: UInt(4) = 0
            b0u: UInt(4) = 0
            b1u: UInt(4) = 0
            s0: UInt(1) = a0[3] ^ b0[3]
            s1: UInt(1) = a1[3] ^ b1[3]
            if a0 < 0:
                a0u = -a0
            else:
                a0u = a0
            if a1 < 0:
                a1u = -a1
            else:
                a1u = a1
            if b0 < 0:
                b0u = -b0
            else:
                b0u = b0
            if b1 < 0:
                b1u = -b1
            else:
                b1u = b1
            op0: UInt(27) = 0
            op1: UInt(18) = 0
            op0[0:4] = a0u
            op0[22:26] = a1u
            op1[0:4] = b0u
            op1[11:15] = b1u
            res: UInt(48) = op0 * op1
            res0u: UInt(8) = res[0:8]
            res1u: UInt(8) = res[33:41]
            res0: int8 = 0
            res1: int8 = 0
            if s0:
                res0 = -res0u
            else:
                res0 = res0u
            if s1:
                res1 = -res1u
            else:
                res1 = res1u
            C[i, j] += res0
            C[i, j] += res1
            A_out[k] = a0
            A_out[k + 1] = a1
            B_out[k] = b0
            B_out[k + 1] = b1

    def systolic_array(A: int4[M, K], B: int4[K, N], C: int8[M, N]):
        A_fifo: int4[M, N + 1, K]
        B_fifo: int4[N, M + 1, K]

        for k in range(K, name="data_load"):
            for m in range(M):
                A_fifo[m, 0, k] = A[m, k]
            for n in range(N):
                B_fifo[n, 0, k] = B[k, n]
        for i, j in allo.grid(M, N, name="PE"):
            kernel(
                A_fifo[i, j], B_fifo[j, i], A_fifo[i, j + 1], B_fifo[j, i + 1], C, i, j
            )
        A_drain: int4[M]
        B_drain: int4[N]
        for k in range(K, name="data_drain"):
            for m in range(M):
                A_drain[m] = A_fifo[m, N, k]
            for n in range(N):
                B_drain[n] = B_fifo[n, M, k]

    s = allo.customize(systolic_array)
    # print(s.module)

    mod = s.build()
    A = np.random.randint(-8, 7, size=(M, K)).astype(np.int8)
    B = np.random.randint(-8, 7, size=(K, N)).astype(np.int8)
    allo_C = np.zeros((M, N), dtype=np.int8)
    mod(A, B, allo_C)
    np_C = A.astype(np.int16) @ B.astype(np.int16)
    np.testing.assert_allclose(allo_C, np_C, atol=1e-3)


def test_subview_systolic_dsp_packed_int4xint8():
    M, N, K = 4, 4, 4
    half_N = 2

    def kernel(
        A_in: int8[K],  # not bit-packed
        B_in: int8[K],  # bit-packed, each element is 4 bits
        A_out: int8[K],
        B_out: int8[K],
        C: int32[M, N],  # bit-packed, each element is 16 bits
        i: index,
        j: index,
    ):
        for k in range(K):
            a: int8 = A_in[k]
            b_packed: int8 = B_in[k]
            b0: int4 = b_packed[0:4]
            b1: int4 = b_packed[4:8]
            au: UInt(8) = 0
            b0u: UInt(4) = 0
            b1u: UInt(4) = 0
            s0: UInt(1) = a[7] ^ b0[3]
            s1: UInt(1) = a[7] ^ b1[3]
            if a < 0:
                au = 0 - a
            else:
                au = a
            if b0 < 0:
                b0u = 0 - b0
            else:
                b0u = b0
            if b1 < 0:
                b1u = 0 - b1
            else:
                b1u = b1
            op0: UInt(18) = 0
            op1: UInt(27) = 0
            op0[0:8] = au
            op1[0:4] = b0u
            op1[13:17] = b1u
            res: UInt(48) = op0 * op1
            res0u: UInt(12) = res[0:12]
            res1u: UInt(12) = res[13:25]
            res0: int16 = 0
            res1: int16 = 0
            if s0:
                res0 = 0 - res0u
            else:
                res0 = res0u
            if s1:
                res1 = 0 - res1u
            else:
                res1 = res1u
            c_packed: int32 = C[i, j]
            c0: int16 = c_packed[0:16]
            c1: int16 = c_packed[16:32]
            c_packed[0:16] = c0 + res0
            c_packed[16:32] = c1 + res1
            C[i, j] = c_packed
            A_out[k] = a
            B_out[k] = b_packed

    def systolic_array(A: int8[M, K], B: int4[K, N], C: int16[M, N]):
        # bitpack B
        B_packed: int8[K, half_N] = 0
        for k in range(K):
            for n in range(half_N):
                B_packed[k, n][0:4] = B[k, n * 2]
                B_packed[k, n][4:8] = B[k, n * 2 + 1]

        A_fifo: int8[M, half_N + 1, K]
        B_fifo: int8[half_N, M + 1, K]

        for k in range(K, name="data_load"):
            for m in range(M):
                A_fifo[m, 0, k] = A[m, k]
            for n in range(half_N):
                B_fifo[n, 0, k] = B_packed[k, n]
        C_packed: int32[M, half_N] = 0
        for i, j in allo.grid(M, half_N, name="PE"):
            kernel(
                A_fifo[i, j],
                B_fifo[j, i],
                A_fifo[i, j + 1],
                B_fifo[j, i + 1],
                C_packed,
                i,
                j,
            )
        A_drain: int8[M]
        B_drain: int8[half_N]
        for k in range(K, name="data_drain"):
            for m in range(M):
                A_drain[m] = A_fifo[m, N, k]
            for n in range(half_N):
                B_drain[n] = B_fifo[n, M, k]
        # unpack C
        for i in range(M):
            for j in range(half_N):
                C[i, j * 2] = C_packed[i, j][0:16]
                C[i, j * 2 + 1] = C_packed[i, j][16:32]

    s = allo.customize(systolic_array)
    # print(s.module)

    mod = s.build()
    A = np.random.randint(-128, 127, size=(M, K)).astype(np.int8)
    B = np.random.randint(-8, 7, size=(K, N)).astype(np.int8)
    np_C = A.astype(np.int16) @ B.astype(np.int16)
    allo_C = np.zeros((M, N), dtype=np.int16)
    mod(A, B, allo_C)
    np.testing.assert_allclose(allo_C, np_C, atol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__])
