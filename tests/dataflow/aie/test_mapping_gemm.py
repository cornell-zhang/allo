# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import int16, int32
import allo.dataflow as df
import numpy as np
from allo.memory import Layout


def _test_gemm_2D_v1():
    LyA = Layout("S0R")
    LyB = Layout("RS1")
    LyC = Layout("S0S1")

    TyI, TyO = int32, int32
    M, N, K = 64, 64, 64
    P0, P1 = 2, 2

    @df.region()
    def top():
        @df.kernel(mapping=[P0, P1])
        def gemm(A: TyI[M, K] @ LyA, B: TyI[K, N] @ LyB, C: TyO[M, N] @ LyC):
            C[:, :] = allo.matmul(A, B)

    mod = df.build(
        top,
        target="aie-mlir",
        mapping_primitives=[
            ("bundle", ["gemm_0_0", "gemm_0_1"]),
            ("bundle", ["gemm_1_0", "gemm_1_1"]),
        ],
    )
    A = np.random.randint(0, 64, (M, K)).astype(np.int32)
    B = np.random.randint(0, 64, (K, N)).astype(np.int32)
    C = np.zeros((M, N)).astype(np.int32)
    mod(A, B, C)
    np.testing.assert_allclose(C, A @ B, atol=1e-5)
    print("PASSED!")


def _test_gemm_2D_v2():
    LyA = Layout("S0R")
    LyB = Layout("RS1")
    LyC = Layout("S0S1")

    TyI, TyO = int32, int32
    M, N, K = 64, 64, 64
    P0, P1 = 2, 2

    @df.region()
    def top():
        @df.kernel(mapping=[P0, P1])
        def gemm(A: TyI[M, K] @ LyA, B: TyI[K, N] @ LyB, C: TyO[M, N] @ LyC):
            C[:, :] = allo.matmul(A, B)

    mod = df.build(
        top,
        target="aie-mlir",
        mapping_primitives=[
            ("chain", ["gemm_0_0", "gemm_0_1"]),
            ("chain", ["gemm_1_0", "gemm_1_1"]),
            ("bundle", ["gemm_0_0-gemm_0_1", "gemm_1_0-gemm_1_1"]),
        ],
    )
    A = np.random.randint(0, 64, (M, K)).astype(np.int32)
    B = np.random.randint(0, 64, (K, N)).astype(np.int32)
    C = np.zeros((M, N)).astype(np.int32)
    mod(A, B, C)
    np.testing.assert_allclose(C, A @ B, atol=1e-5)
    print("PASSED!")


def _test_pingpong_gemm_2x2x2():

    Ty = int16
    M, N, K = 32, 32, 32
    Pm, Pn, Pk = 2, 2, 2
    Mt, Nt, Kt = M // Pm, N // Pn, K // Pk

    LyA = Layout("S1S2")
    LyB = Layout("S2S0")
    LyC = Layout("S1S0")

    @df.region()
    def top():
        pipe = df.array(
            df.pipe(dtype=Ty, shape=(Mt, Nt), depth=2), shape=(Pk - 1, Pm, Pn)
        )

        @df.kernel(mapping=[Pk, Pm, Pn])
        def gemm(A: Ty[M, K] @ LyA, B: Ty[K, N] @ LyB, C: Ty[M, N] @ LyC):
            pk, pm, pn = df.get_pid()
            with allo.meta_if(pk > 0):
                C_in: Ty[Mt, Nt] = pipe[pk - 1, pm, pn].get()
            with allo.meta_else():
                C_in: Ty[Mt, Nt] = 0
            C_out: Ty[Mt, Nt] = allo.add(allo.matmul(A, B), C_in)
            with allo.meta_if(pk < Pk - 1):
                pipe[pk, pm, pn].put(C_out)
            with allo.meta_elif(pk == Pk - 1):
                C[:, :] = C_out

    mod = df.build(
        top,
        target="aie-mlir",
        mapping_primitives=[
            ("chain", ["gemm_0_0_0", "gemm_1_0_0"]),
            ("chain", ["gemm_0_0_1", "gemm_1_0_1"]),
            ("chain", ["gemm_0_1_0", "gemm_1_1_0"]),
            ("chain", ["gemm_0_1_1", "gemm_1_1_1"]),
        ],
    )
    A = np.random.randint(0, 64, (M, K)).astype(np.int16)
    B = np.random.randint(0, 64, (K, N)).astype(np.int16)
    C = np.zeros((M, N)).astype(np.int16)
    mod(A, B, C)
    np.testing.assert_allclose(C, A @ B, atol=1e-5)
    print("PASSED!")


def _test_pingpong_gemm_2x2x2_partial_chain():

    Ty = int16
    M, N, K = 32, 32, 32
    Pm, Pn, Pk = 2, 2, 2
    Mt, Nt, Kt = M // Pm, N // Pn, K // Pk

    LyA = Layout("S1S2")
    LyB = Layout("S2S0")
    LyC = Layout("S1S0")

    @df.region()
    def top():
        pipe = df.array(
            df.pipe(dtype=Ty, shape=(Mt, Nt), depth=2), shape=(Pk - 1, Pm, Pn)
        )

        @df.kernel(mapping=[Pk, Pm, Pn])
        def gemm(A: Ty[M, K] @ LyA, B: Ty[K, N] @ LyB, C: Ty[M, N] @ LyC):
            pk, pm, pn = df.get_pid()
            with allo.meta_if(pk > 0):
                C_in: Ty[Mt, Nt] = pipe[pk - 1, pm, pn].get()
            with allo.meta_else():
                C_in: Ty[Mt, Nt] = 0
            C_out: Ty[Mt, Nt] = allo.add(allo.matmul(A, B), C_in)
            with allo.meta_if(pk < Pk - 1):
                pipe[pk, pm, pn].put(C_out)
            with allo.meta_elif(pk == Pk - 1):
                C[:, :] = C_out

    mod = df.build(
        top,
        target="aie-mlir",
        mapping_primitives=[
            ("chain", ["gemm_0_0_0", "gemm_1_0_0"]),
            ("chain", ["gemm_0_0_1", "gemm_1_0_1"]),
        ],
    )
    A = np.random.randint(0, 64, (M, K)).astype(np.int16)
    B = np.random.randint(0, 64, (K, N)).astype(np.int16)
    C = np.zeros((M, N)).astype(np.int16)
    mod(A, B, C)
    np.testing.assert_allclose(C, A @ B, atol=1e-5)
    print("PASSED!")


def _test_pingpong_gemm_1x1x4():

    Ty = int16
    M, N, K = 32, 32, 128
    Pm, Pn, Pk = 1, 1, 4
    Mt, Nt, Kt = M // Pm, N // Pn, K // Pk

    LyA = Layout("S1S2")
    LyB = Layout("S2S0")
    LyC = Layout("S1S0")

    @df.region()
    def top():
        pipe = df.array(
            df.pipe(dtype=Ty, shape=(Mt, Nt), depth=2), shape=(Pk - 1, Pm, Pn)
        )

        @df.kernel(mapping=[Pk, Pm, Pn])
        def gemm(A: Ty[M, K] @ LyA, B: Ty[K, N] @ LyB, C: Ty[M, N] @ LyC):
            pk, pm, pn = df.get_pid()
            with allo.meta_if(pk > 0):
                C_in: Ty[Mt, Nt] = pipe[pk - 1, pm, pn].get()
            with allo.meta_else():
                C_in: Ty[Mt, Nt] = 0
            C_out: Ty[Mt, Nt] = allo.add(allo.matmul(A, B), C_in)
            with allo.meta_if(pk < Pk - 1):
                pipe[pk, pm, pn].put(C_out)
            with allo.meta_elif(pk == Pk - 1):
                C[:, :] = C_out

    mod = df.build(
        top,
        target="aie-mlir",
        mapping_primitives=[
            ("chain", ["gemm_0_0_0", "gemm_1_0_0"]),
            ("chain", ["gemm_0_0_0-gemm_1_0_0", "gemm_2_0_0"]),
            ("chain", ["gemm_0_0_0-gemm_1_0_0-gemm_2_0_0", "gemm_3_0_0"]),
        ],
    )
    A = np.random.randint(0, 64, (M, K)).astype(np.int16)
    B = np.random.randint(0, 64, (K, N)).astype(np.int16)
    C = np.zeros((M, N)).astype(np.int16)
    mod(A, B, C)
    np.testing.assert_allclose(C, A @ B, atol=1e-5)
    print("PASSED!")


def _test_pingpong_gemm_2x2x4():

    Ty = int16
    M, N, K = 64, 64, 128
    Pm, Pn, Pk = 2, 2, 4
    Mt, Nt, Kt = M // Pm, N // Pn, K // Pk

    LyA = Layout("S1S2")
    LyB = Layout("S2S0")
    LyC = Layout("S1S0")

    @df.region()
    def top():
        pipe = df.array(
            df.pipe(dtype=Ty, shape=(Mt, Nt), depth=2), shape=(Pk - 1, Pm, Pn)
        )

        @df.kernel(mapping=[Pk, Pm, Pn])
        def gemm(A: Ty[M, K] @ LyA, B: Ty[K, N] @ LyB, C: Ty[M, N] @ LyC):
            pk, pm, pn = df.get_pid()
            with allo.meta_if(pk > 0):
                C_in: Ty[Mt, Nt] = pipe[pk - 1, pm, pn].get()
            with allo.meta_else():
                C_in: Ty[Mt, Nt] = 0
            C_out: Ty[Mt, Nt] = allo.add(allo.matmul(A, B), C_in)
            with allo.meta_if(pk < Pk - 1):
                pipe[pk, pm, pn].put(C_out)
            with allo.meta_elif(pk == Pk - 1):
                C[:, :] = C_out

    mod = df.build(
        top,
        target="aie-mlir",
        mapping_primitives=[
            ("chain", ["gemm_0_0_0", "gemm_1_0_0"]),
            ("chain", ["gemm_0_0_0-gemm_1_0_0", "gemm_2_0_0"]),
            ("chain", ["gemm_0_0_0-gemm_1_0_0-gemm_2_0_0", "gemm_3_0_0"]),
            ("chain", ["gemm_0_0_1", "gemm_1_0_1"]),
            ("chain", ["gemm_0_0_1-gemm_1_0_1", "gemm_2_0_1"]),
            ("chain", ["gemm_0_0_1-gemm_1_0_1-gemm_2_0_1", "gemm_3_0_1"]),
            ("chain", ["gemm_0_1_0", "gemm_1_1_0"]),
            ("chain", ["gemm_0_1_0-gemm_1_1_0", "gemm_2_1_0"]),
            ("chain", ["gemm_0_1_0-gemm_1_1_0-gemm_2_1_0", "gemm_3_1_0"]),
            ("chain", ["gemm_0_1_1", "gemm_1_1_1"]),
            ("chain", ["gemm_0_1_1-gemm_1_1_1", "gemm_2_1_1"]),
            ("chain", ["gemm_0_1_1-gemm_1_1_1-gemm_2_1_1", "gemm_3_1_1"]),
        ],
    )
    A = np.random.randint(0, 64, (M, K)).astype(np.int16)
    B = np.random.randint(0, 64, (K, N)).astype(np.int16)
    C = np.zeros((M, N)).astype(np.int16)
    mod(A, B, C)
    np.testing.assert_allclose(C, A @ B, atol=1e-5)
    print("PASSED!")


def _test_pingpong_gemm_4x4x4():

    Ty = int16
    M, N, K = 128, 128, 128
    Pm, Pn, Pk = 4, 4, 4
    Mt, Nt, Kt = M // Pm, N // Pn, K // Pk

    LyA = Layout("S1S2")
    LyB = Layout("S2S0")
    LyC = Layout("S1S0")

    @df.region()
    def top():
        pipe = df.array(
            df.pipe(dtype=Ty, shape=(Mt, Nt), depth=2), shape=(Pk - 1, Pm, Pn)
        )

        @df.kernel(mapping=[Pk, Pm, Pn])
        def gemm(A: Ty[M, K] @ LyA, B: Ty[K, N] @ LyB, C: Ty[M, N] @ LyC):
            pk, pm, pn = df.get_pid()
            with allo.meta_if(pk > 0):
                C_in: Ty[Mt, Nt] = pipe[pk - 1, pm, pn].get()
            with allo.meta_else():
                C_in: Ty[Mt, Nt] = 0
            C_out: Ty[Mt, Nt] = allo.add(allo.matmul(A, B), C_in)
            with allo.meta_if(pk < Pk - 1):
                pipe[pk, pm, pn].put(C_out)
            with allo.meta_elif(pk == Pk - 1):
                C[:, :] = C_out

    mod = df.build(
        top,
        target="aie-mlir",
        mapping_primitives=[
            ("chain", ["gemm_0_0_0", "gemm_1_0_0"]),
            ("chain", ["gemm_0_0_0-gemm_1_0_0", "gemm_2_0_0"]),
            ("chain", ["gemm_0_0_0-gemm_1_0_0-gemm_2_0_0", "gemm_3_0_0"]),
            ("chain", ["gemm_0_0_1", "gemm_1_0_1"]),
            ("chain", ["gemm_0_0_1-gemm_1_0_1", "gemm_2_0_1"]),
            ("chain", ["gemm_0_0_1-gemm_1_0_1-gemm_2_0_1", "gemm_3_0_1"]),
            ("chain", ["gemm_0_0_2", "gemm_1_0_2"]),
            ("chain", ["gemm_0_0_2-gemm_1_0_2", "gemm_2_0_2"]),
            ("chain", ["gemm_0_0_2-gemm_1_0_2-gemm_2_0_2", "gemm_3_0_2"]),
            ("chain", ["gemm_0_0_3", "gemm_1_0_3"]),
            ("chain", ["gemm_0_0_3-gemm_1_0_3", "gemm_2_0_3"]),
            ("chain", ["gemm_0_0_3-gemm_1_0_3-gemm_2_0_3", "gemm_3_0_3"]),
            ("chain", ["gemm_0_1_0", "gemm_1_1_0"]),
            ("chain", ["gemm_0_1_0-gemm_1_1_0", "gemm_2_1_0"]),
            ("chain", ["gemm_0_1_0-gemm_1_1_0-gemm_2_1_0", "gemm_3_1_0"]),
            ("chain", ["gemm_0_1_1", "gemm_1_1_1"]),
            ("chain", ["gemm_0_1_1-gemm_1_1_1", "gemm_2_1_1"]),
            ("chain", ["gemm_0_1_1-gemm_1_1_1-gemm_2_1_1", "gemm_3_1_1"]),
            ("chain", ["gemm_0_1_2", "gemm_1_1_2"]),
            ("chain", ["gemm_0_1_2-gemm_1_1_2", "gemm_2_1_2"]),
            ("chain", ["gemm_0_1_2-gemm_1_1_2-gemm_2_1_2", "gemm_3_1_2"]),
            ("chain", ["gemm_0_1_3", "gemm_1_1_3"]),
            ("chain", ["gemm_0_1_3-gemm_1_1_3", "gemm_2_1_3"]),
            ("chain", ["gemm_0_1_3-gemm_1_1_3-gemm_2_1_3", "gemm_3_1_3"]),
            ("chain", ["gemm_0_2_0", "gemm_1_2_0"]),
            ("chain", ["gemm_0_2_0-gemm_1_2_0", "gemm_2_2_0"]),
            ("chain", ["gemm_0_2_0-gemm_1_2_0-gemm_2_2_0", "gemm_3_2_0"]),
            ("chain", ["gemm_0_2_1", "gemm_1_2_1"]),
            ("chain", ["gemm_0_2_1-gemm_1_2_1", "gemm_2_2_1"]),
            ("chain", ["gemm_0_2_1-gemm_1_2_1-gemm_2_2_1", "gemm_3_2_1"]),
            ("chain", ["gemm_0_2_2", "gemm_1_2_2"]),
            ("chain", ["gemm_0_2_2-gemm_1_2_2", "gemm_2_2_2"]),
            ("chain", ["gemm_0_2_2-gemm_1_2_2-gemm_2_2_2", "gemm_3_2_2"]),
            ("chain", ["gemm_0_2_3", "gemm_1_2_3"]),
            ("chain", ["gemm_0_2_3-gemm_1_2_3", "gemm_2_2_3"]),
            ("chain", ["gemm_0_2_3-gemm_1_2_3-gemm_2_2_3", "gemm_3_2_3"]),
            ("chain", ["gemm_0_3_0", "gemm_1_3_0"]),
            ("chain", ["gemm_0_3_0-gemm_1_3_0", "gemm_2_3_0"]),
            ("chain", ["gemm_0_3_0-gemm_1_3_0-gemm_2_3_0", "gemm_3_3_0"]),
            ("chain", ["gemm_0_3_1", "gemm_1_3_1"]),
            ("chain", ["gemm_0_3_1-gemm_1_3_1", "gemm_2_3_1"]),
            ("chain", ["gemm_0_3_1-gemm_1_3_1-gemm_2_3_1", "gemm_3_3_1"]),
            ("chain", ["gemm_0_3_2", "gemm_1_3_2"]),
            ("chain", ["gemm_0_3_2-gemm_1_3_2", "gemm_2_3_2"]),
            ("chain", ["gemm_0_3_2-gemm_1_3_2-gemm_2_3_2", "gemm_3_3_2"]),
            ("chain", ["gemm_0_3_3", "gemm_1_3_3"]),
            ("chain", ["gemm_0_3_3-gemm_1_3_3", "gemm_2_3_3"]),
            ("chain", ["gemm_0_3_3-gemm_1_3_3-gemm_2_3_3", "gemm_3_3_3"]),
        ],
    )
    A = np.random.randint(0, 64, (M, K)).astype(np.int16)
    B = np.random.randint(0, 64, (K, N)).astype(np.int16)
    C = np.zeros((M, N)).astype(np.int16)
    mod(A, B, C)
    np.testing.assert_allclose(C, A @ B, atol=1e-5)
    print("PASSED!")


if __name__ == "__main__":
    _test_gemm_2D_v1()
    _test_gemm_2D_v2()
    _test_pingpong_gemm_2x2x2()
    _test_pingpong_gemm_2x2x2_partial_chain()
    _test_pingpong_gemm_1x1x4()
    _test_pingpong_gemm_2x2x4()
    _test_pingpong_gemm_4x4x4()
