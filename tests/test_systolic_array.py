# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import numpy as np
import pytest
import allo
from allo.ir.types import int4, int8, int16, int32, index, UInt
from allo.ir.utils import MockBuffer


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


def test_subview_systolic_stream():
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
    s.partition(s.C, dim=0)  # required, otherwise it will fail dataflow checking
    s.partition(s.A, dim=1)
    s.partition(s.B, dim=2)
    pe = s.unfold("PE", [0, 1])  # specify which are spatial loops
    s.to(s.A_fifo, pe, axis=1, depth=M + 1)
    s.to(s.B_fifo, pe, axis=0, depth=N + 1)
    code = s.build("vhls")
    assert "#pragma HLS dataflow" in str(code)
    if os.system("which vivado_hls >> /dev/null") == 0:
        hls_mod = s.build(
            target="vivado_hls", mode="debug", project="systolic_stream.prj"
        )
        print(hls_mod)
        hls_mod()


def test_parameterized_systolic():
    from allo.library.systolic import systolic_tile

    s = allo.customize(
        systolic_tile,
        instantiate={"T_A": int8, "T_B": int8, "T_C": int16, "Mt": 4, "Nt": 4, "K": 4},
    )
    print(s.module)
    mod = s.build()
    M, N, K = 4, 4, 4
    A = np.random.randint(-8, 8, size=(M, K)).astype(np.int8)
    B = np.random.randint(-8, 8, size=(K, N)).astype(np.int8)
    allo_C = np.zeros((M, N), dtype=np.int16)
    mod(A, B, allo_C)
    np_C = A.astype(np.int16) @ B.astype(np.int16)
    np.testing.assert_allclose(allo_C, np_C, atol=1e-3)
    print("Passed!")


def test_tiled_systolic():
    from allo.library.systolic import systolic

    MM, KK, NN = 8, 16, 8
    s = allo.customize(
        systolic,
        instantiate={
            "T_A": int8,
            "T_B": int8,
            "T_C": int16,
            "M": MM,
            "N": NN,
            "K": KK,
            "Mt": 4,
            "Nt": 4,
        },
    )
    print(s.module)
    mod = s.build()
    A = np.random.randint(-4, 4, size=(MM, KK)).astype(np.int8)
    B = np.random.randint(-4, 4, size=(KK, NN)).astype(np.int8)
    allo_C = np.zeros((MM, NN), dtype=np.int16)
    mod(A, B, allo_C)
    np_C = A.astype(np.int16) @ B.astype(np.int16)
    np.testing.assert_allclose(allo_C, np_C, atol=1e-3)
    print("Passed!")


def test_cascade_systolic():
    from allo.library.systolic import systolic_tile

    M0, M1, KK = 4, 4, 4
    W_A_cst = np.random.randint(-4, 4, size=(M0, M1)).astype(np.int8)
    W_B_cst = np.random.randint(-4, 4, size=(M0, M1)).astype(np.int8)

    def top(X: int8[M0, M1]) -> int8[M0, M1]:
        Z: int8[M0, M1] = 0
        Y: int8[M0, M1] = 0
        W_A: int8[M0, M1] = W_A_cst
        W_B: int8[M0, M1] = W_B_cst
        systolic_tile(X, W_A, Z)
        systolic_tile(Z, W_B, Y)
        return Y

    s_top = allo.customize(
        top,
        instantiate={
            "T_A": int8,
            "T_B": int8,
            "T_C": int8,
            "Mt": M0,
            "Nt": M1,
            "K": KK,
        },
    )
    # CPU testing
    mod = s_top.build()
    X = np.random.randint(-4, 4, size=(M0, M1)).astype(np.int8)
    allo_C = mod(X)
    np_C = X @ W_A_cst @ W_B_cst
    np.testing.assert_allclose(allo_C, np_C, atol=1e-3)
    print("Passed!")
    # Submodule customization
    s = allo.customize(
        systolic_tile,
        instantiate={
            "T_A": int8,
            "T_B": int8,
            "T_C": int8,
            "Mt": M0,
            "Nt": M1,
            "K": KK,
        },
    )
    s.partition(s.C, dim=0)  # required, otherwise it will fail dataflow checking
    s.partition(s.A, dim=1)
    s.partition(s.B, dim=2)
    pe = s.unfold("PE", [0, 1])  # specify which are spatial loops
    s.to(s.A_fifo, pe, axis=1, depth=M0 + 1)
    s.to(s.B_fifo, pe, axis=0, depth=M1 + 1)
    code = s.build("vhls")
    # Compose with submodule
    s_top.compose(s)
    # HLS testing
    code = s_top.build("vhls")
    print(code)
    if os.system("which vitis_hls >> /dev/null") == 0:
        hls_mod = s_top.build(
            target="vitis_hls", mode="hw_emu", project=f"sa_{M0}x{M1}.prj"
        )
        hls_mod()


def test_cascade_specialized_systolic():
    from allo.library.systolic import systolic

    M0, M1 = 2, 2
    MM, NN, KK = 4, 4, 4
    W_A_cst = np.random.randint(-4, 4, size=(MM, NN)).astype(np.int8)
    W_B_cst = np.random.randint(-4, 4, size=(MM, NN)).astype(np.int8)

    def top(X: int8[MM, NN]) -> int8[MM, NN]:
        Z: int8[MM, NN]  # will become FIFO later, so don't need to initialize
        Y: int8[MM, NN]
        W_A: int8[MM, NN] = W_A_cst
        W_B: int8[MM, NN] = W_B_cst
        systolic["FFN1"](X, W_A, Z)
        systolic["FFN2"](Z, W_B, Y)
        return Y

    s_top = allo.customize(
        top,
        instantiate={
            "T_A": int8,
            "T_B": int8,
            "T_C": int8,
            "Mt": M0,
            "Nt": M1,
            "M": MM,
            "N": NN,
            "K": KK,
        },
    )
    print(s_top.module)
    # CPU testing
    mod = s_top.build()
    X = np.random.randint(-4, 4, size=(MM, NN)).astype(np.int8)
    allo_C = mod(X)
    np_C = X @ W_A_cst @ W_B_cst
    np.testing.assert_allclose(allo_C, np_C, atol=1e-3)
    print("Passed!")
    # Submodule customization
    s = allo.customize(
        systolic,
        instantiate={
            "T_A": int8,
            "T_B": int8,
            "T_C": int8,
            "Mt": M0,
            "Nt": M1,
            "M": MM,
            "N": NN,
            "K": KK,
        },
    )
    s.partition(s.local_C, dim=0)  # required, otherwise it will fail dataflow checking
    s.partition(s.local_A, dim=1)
    s.partition(s.local_B, dim=2)
    load_A_loop = s.get_loops("systolic")["outer_tile"]["ai"]
    s.pipeline(load_A_loop)
    load_B_loop = s.get_loops("systolic")["outer_tile"]["bj"]
    s.pipeline(load_B_loop)
    store_C_loop = s.get_loops("systolic")["outer_tile"]["si"]
    s.pipeline(store_C_loop)
    tile_loop = s.get_loops("systolic")["outer_tile"]["ni"]
    s.dataflow(tile_loop)
    pe = s.unfold("systolic_tile:PE", [0, 1])  # specify which are spatial loops
    s.to(MockBuffer("systolic_tile", "A_fifo"), pe, axis=1, depth=M0 + 1)
    s.to(MockBuffer("systolic_tile", "B_fifo"), pe, axis=0, depth=M1 + 1)
    # Compose with submodule
    s_top.use_def_chain.dump_graph("top")
    s_top.compose(s, id="FFN1")
    s_top.compose(s, id="FFN2")
    s_top.to(s_top.Z, "systolic_FFN2", depth=M0 * KK)
    # HLS testing
    code = s_top.build("vhls")
    print(code)
    if os.system("which vitis_hls >> /dev/null") == 0:
        hls_mod = s_top.build(
            target="vitis_hls",
            mode="hw_emu",
            project=f"sa_{MM}x{NN}_tile_{M0}x{M1}.prj",
        )
        hls_mod()


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
            s0: UInt(1) = a0[3] ^ b0[3]
            s1: UInt(1) = a1[3] ^ b1[3]
            a0u: UInt(4) = -a0 if a0 < 0 else a0
            a1u: UInt(4) = -a1 if a1 < 0 else a1
            b0u: UInt(4) = -b0 if b0 < 0 else b0
            b1u: UInt(4) = -b1 if b1 < 0 else b1
            op0: UInt(27) = 0
            op1: UInt(18) = 0
            op0[0:4] = a0u
            op0[22:26] = a1u
            op1[0:4] = b0u
            op1[11:15] = b1u
            res: UInt(48) = op0 * op1
            res0u: UInt(8) = res[0:8]
            res1u: UInt(8) = res[33:41]
            res0: int8 = -res0u if s0 else res0u
            res1: int8 = -res1u if s1 else res1u
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
    print(s.module)

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
            s0: UInt(1) = a[7] ^ b0[3]
            s1: UInt(1) = a[7] ^ b1[3]
            au: UInt(8) = allo.abs(a)
            b0u: UInt(4) = allo.abs(b0)
            b1u: UInt(4) = allo.abs(b1)
            op0: UInt(18) = 0
            op1: UInt(27) = 0
            op0[0:8] = au
            op1[0:4] = b0u
            op1[13:17] = b1u
            res: UInt(48) = op0 * op1
            res0u: UInt(12) = res[0:12]
            res1u: UInt(12) = res[13:25]
            res0: int16 = -res0u if s0 else res0u
            res1: int16 = -res1u if s1 else res1u
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
