# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.utils import construct_kernel_name
from allo.ir.types import int16, int32, Stream
import allo.dataflow as df
import numpy as np
from allo.memory import Layout
from allo.backend.aie import is_available


def _gemm(*coordinates):
    return construct_kernel_name("gemm", coordinates)


def _partial_gemm(*coordinates):
    return construct_kernel_name("partial_gemm", coordinates)


def _chain_name(*coordinates):
    """Name an ordered chain of GEMM PEs selected by coordinates."""
    return "-".join(_gemm(*point) for point in coordinates)


S = Layout.Shard
R = Layout.Replicate


def test_gemm_2D_v1():
    LyA = [S(1), R]
    LyB = [R, S(0)]
    LyC = [S(1), S(0)]

    TyI, TyO = int32, int32
    M, N, K = 64, 64, 64
    P0, P1 = 2, 2

    @df.region()
    def top(A: TyI[M, K], B: TyI[K, N], C: TyO[M, N]):
        @df.kernel(mapping=[P0, P1], args=[A, B, C])
        def gemm(
            local_A: TyI[M, K] @ LyA, local_B: TyI[K, N] @ LyB, local_C: TyO[M, N] @ LyC
        ):
            local_C[:, :] = allo.matmul(local_A, local_B)

    if is_available():
        mod = df.build(
            top,
            target="aie",
            mapping_primitives=[
                ("bundle", [_gemm(0, 0), _gemm(0, 1)]),
                ("bundle", [_gemm(1, 0), _gemm(1, 1)]),
            ],
        )
        A = np.random.randint(0, 64, (M, K)).astype(np.int32)
        B = np.random.randint(0, 64, (K, N)).astype(np.int32)
        C = np.zeros((M, N)).astype(np.int32)
        mod(A, B, C)
        np.testing.assert_allclose(C, A @ B, atol=1e-5)
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")

    mod = df.build(
        top,
        target="aie",
        mapping_primitives=[
            ("bundle", [(_gemm(0, 0), _gemm(1, 0)), (_gemm(0, 1), _gemm(1, 1))])
        ],
    )
    C = np.zeros((M, N)).astype(np.int32)
    mod(A, B, C)
    np.testing.assert_allclose(C, A @ B, atol=1e-5)
    print("PASSED!")


def test_gemm_2D_v2():
    LyA = [S(1), R]
    LyB = [R, S(0)]
    LyC = [S(1), S(0)]

    TyI, TyO = int32, int32
    M, N, K = 64, 64, 64
    P0, P1 = 2, 2

    @df.region()
    def top(A: TyI[M, K], B: TyI[K, N], C: TyO[M, N]):
        @df.kernel(mapping=[P0, P1], args=[A, B, C])
        def gemm(
            local_A: TyI[M, K] @ LyA, local_B: TyI[K, N] @ LyB, local_C: TyO[M, N] @ LyC
        ):
            local_C[:, :] = allo.matmul(local_A, local_B)

    if is_available():
        mod = df.build(
            top,
            target="aie",
            mapping_primitives=[
                ("chain", [_gemm(0, 0), _gemm(0, 1)]),
                ("chain", [_gemm(1, 0), _gemm(1, 1)]),
                ("bundle", [_chain_name((0, 0), (0, 1)), _chain_name((1, 0), (1, 1))]),
            ],
        )
        A = np.random.randint(0, 64, (M, K)).astype(np.int32)
        B = np.random.randint(0, 64, (K, N)).astype(np.int32)
        C = np.zeros((M, N)).astype(np.int32)
        mod(A, B, C)
        np.testing.assert_allclose(C, A @ B, atol=1e-5)
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


def test_pingpong_gemm_2x2x2():

    Ty = int16
    M, N, K = 32, 32, 32
    Pm, Pn, Pk = 2, 2, 2
    Mt, Nt, Kt = M // Pm, N // Pn, K // Pk

    LyA = [S(1), S(0)]
    LyB = [S(0), S(2)]
    LyC = [S(1), S(2)]

    @df.region()
    def top(A: Ty[M, K], B: Ty[K, N], C: Ty[M, N]):
        pipe: Stream[Ty[Mt, Nt], 2][Pk - 1, Pm, Pn]

        @df.kernel(mapping=[Pk, Pm, Pn], args=[A, B, C])
        def gemm(
            local_A: Ty[M, K] @ LyA, local_B: Ty[K, N] @ LyB, local_C: Ty[M, N] @ LyC
        ):
            pk, pm, pn = df.get_pid()
            C_in: Ty[Mt, Nt]
            with allo.meta_if(pk > 0):
                C_in[:, :] = pipe[pk - 1, pm, pn].get()
            with allo.meta_else():
                C_in[:, :] = 0
            C_out: Ty[Mt, Nt] = allo.add(allo.matmul(local_A, local_B), C_in)
            with allo.meta_if(pk < Pk - 1):
                pipe[pk, pm, pn].put(C_out)
            with allo.meta_elif(pk == Pk - 1):
                local_C[:, :] = C_out

    if is_available():
        mod = df.build(
            top,
            target="aie",
            mapping_primitives=[
                ("chain", [_gemm(0, 0, 0), _gemm(1, 0, 0)]),
                ("chain", [_gemm(0, 0, 1), _gemm(1, 0, 1)]),
                ("chain", [_gemm(0, 1, 0), _gemm(1, 1, 0)]),
                ("chain", [_gemm(0, 1, 1), _gemm(1, 1, 1)]),
            ],
        )
        A = np.random.randint(0, 64, (M, K)).astype(np.int16)
        B = np.random.randint(0, 64, (K, N)).astype(np.int16)
        C = np.zeros((M, N)).astype(np.int16)
        mod(A, B, C)
        np.testing.assert_allclose(C, A @ B, atol=1e-5)
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")

    mod = df.build(
        top,
        target="aie",
        mapping_primitives=[
            (
                "bundle",
                [
                    (_gemm(0, 0, 0), _gemm(1, 0, 0)),
                    (_gemm(0, 0, 1), _gemm(1, 0, 1)),
                    (_gemm(0, 1, 0), _gemm(1, 1, 0)),
                    (_gemm(0, 1, 1), _gemm(1, 1, 1)),
                ],
            )
        ],
    )
    A = np.random.randint(0, 64, (M, K)).astype(np.int16)
    B = np.random.randint(0, 64, (K, N)).astype(np.int16)
    C = np.zeros((M, N)).astype(np.int16)
    mod(A, B, C)
    np.testing.assert_allclose(C, A @ B, atol=1e-5)
    print("PASSED!")


def test_pingpong_gemm_2x2x2_partial_chain():

    Ty = int16
    M, N, K = 32, 32, 32
    Pm, Pn, Pk = 2, 2, 2
    Mt, Nt, Kt = M // Pm, N // Pn, K // Pk

    LyA = [S(1), S(0)]
    LyB = [S(0), S(2)]
    LyC = [S(1), S(2)]

    @df.region()
    def top(A: Ty[M, K], B: Ty[K, N], C: Ty[M, N]):
        pipe: Stream[Ty[Mt, Nt], 2][Pk - 1, Pm, Pn]

        @df.kernel(mapping=[Pk, Pm, Pn], args=[A, B, C])
        def gemm(
            local_A: Ty[M, K] @ LyA, local_B: Ty[K, N] @ LyB, local_C: Ty[M, N] @ LyC
        ):
            pk, pm, pn = df.get_pid()
            C_in: Ty[Mt, Nt]
            with allo.meta_if(pk > 0):
                C_in[:, :] = pipe[pk - 1, pm, pn].get()
            with allo.meta_else():
                C_in[:, :] = 0
            C_out: Ty[Mt, Nt] = allo.add(allo.matmul(local_A, local_B), C_in)
            with allo.meta_if(pk < Pk - 1):
                pipe[pk, pm, pn].put(C_out)
            with allo.meta_elif(pk == Pk - 1):
                local_C[:, :] = C_out

    if is_available():
        mod = df.build(
            top,
            target="aie",
            mapping_primitives=[
                ("chain", [_gemm(0, 0, 0), _gemm(1, 0, 0)]),
                ("chain", [_gemm(0, 0, 1), _gemm(1, 0, 1)]),
            ],
        )
        A = np.random.randint(0, 64, (M, K)).astype(np.int16)
        B = np.random.randint(0, 64, (K, N)).astype(np.int16)
        C = np.zeros((M, N)).astype(np.int16)
        mod(A, B, C)
        np.testing.assert_allclose(C, A @ B, atol=1e-5)
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


def test_pingpong_gemm_1x1x4():

    Ty = int16
    M, N, K = 32, 32, 128
    Pm, Pn, Pk = 1, 1, 4
    Mt, Nt, Kt = M // Pm, N // Pn, K // Pk

    LyA = [S(1), S(0)]
    LyB = [S(0), S(2)]
    LyC = [S(1), S(2)]

    @df.region()
    def top(A: Ty[M, K], B: Ty[K, N], C: Ty[M, N]):
        pipe: Stream[Ty[Mt, Nt], 2][Pk - 1, Pm, Pn]

        @df.kernel(mapping=[Pk, Pm, Pn], args=[A, B, C])
        def gemm(
            local_A: Ty[M, K] @ LyA, local_B: Ty[K, N] @ LyB, local_C: Ty[M, N] @ LyC
        ):
            pk, pm, pn = df.get_pid()
            C_in: Ty[Mt, Nt]
            with allo.meta_if(pk > 0):
                C_in[:, :] = pipe[pk - 1, pm, pn].get()
            with allo.meta_else():
                C_in[:, :] = 0
            C_out: Ty[Mt, Nt] = allo.add(allo.matmul(local_A, local_B), C_in)
            with allo.meta_if(pk < Pk - 1):
                pipe[pk, pm, pn].put(C_out)
            with allo.meta_elif(pk == Pk - 1):
                local_C[:, :] = C_out

    A = np.random.randint(0, 64, (M, K)).astype(np.int16)
    B = np.random.randint(0, 64, (K, N)).astype(np.int16)
    C = np.zeros((M, N)).astype(np.int16)

    if is_available():
        mod_v1 = df.build(
            top,
            target="aie",
            mapping_primitives=[
                ("chain", [_gemm(0, 0, 0), _gemm(1, 0, 0)]),
                ("chain", [_chain_name((0, 0, 0), (1, 0, 0)), _gemm(2, 0, 0)]),
                (
                    "chain",
                    [_chain_name((0, 0, 0), (1, 0, 0), (2, 0, 0)), _gemm(3, 0, 0)],
                ),
            ],
        )
        mod_v1(A, B, C)
        np.testing.assert_allclose(C, A @ B, atol=1e-5)
        print("V1 PASSED!")

        mod_v2 = df.build(
            top,
            target="aie",
            mapping_primitives=[
                ("chain", [_gemm(0, 0, 0), _gemm(1, 0, 0)]),
                ("chain", [_gemm(2, 0, 0), _gemm(3, 0, 0)]),
            ],
        )
        mod_v2(A, B, C)
        np.testing.assert_allclose(C, A @ B, atol=1e-5)
        print("V2 PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


def test_pingpong_gemm_2x2x4():

    Ty = int16
    M, N, K = 64, 64, 128
    Pm, Pn, Pk = 2, 2, 4
    Mt, Nt, Kt = M // Pm, N // Pn, K // Pk

    LyA = [S(1), S(0)]
    LyB = [S(0), S(2)]
    LyC = [S(1), S(2)]

    @df.region()
    def top(A: Ty[M, K], B: Ty[K, N], C: Ty[M, N]):
        pipe: Stream[Ty[Mt, Nt], 2][Pk - 1, Pm, Pn]

        @df.kernel(mapping=[Pk, Pm, Pn], args=[A, B, C])
        def gemm(
            local_A: Ty[M, K] @ LyA, local_B: Ty[K, N] @ LyB, local_C: Ty[M, N] @ LyC
        ):
            pk, pm, pn = df.get_pid()
            C_in: Ty[Mt, Nt]
            with allo.meta_if(pk > 0):
                C_in[:, :] = pipe[pk - 1, pm, pn].get()
            with allo.meta_else():
                C_in[:, :] = 0
            C_out: Ty[Mt, Nt] = allo.add(allo.matmul(local_A, local_B), C_in)
            with allo.meta_if(pk < Pk - 1):
                pipe[pk, pm, pn].put(C_out)
            with allo.meta_elif(pk == Pk - 1):
                local_C[:, :] = C_out

    A = np.random.randint(0, 64, (M, K)).astype(np.int16)
    B = np.random.randint(0, 64, (K, N)).astype(np.int16)
    C = np.zeros((M, N)).astype(np.int16)
    if is_available():
        mod = df.build(
            top,
            target="aie",
            mapping_primitives=[
                ("chain", [_gemm(0, 0, 0), _gemm(1, 0, 0)]),
                ("chain", [_chain_name((0, 0, 0), (1, 0, 0)), _gemm(2, 0, 0)]),
                (
                    "chain",
                    [_chain_name((0, 0, 0), (1, 0, 0), (2, 0, 0)), _gemm(3, 0, 0)],
                ),
                ("chain", [_gemm(0, 0, 1), _gemm(1, 0, 1)]),
                ("chain", [_chain_name((0, 0, 1), (1, 0, 1)), _gemm(2, 0, 1)]),
                (
                    "chain",
                    [_chain_name((0, 0, 1), (1, 0, 1), (2, 0, 1)), _gemm(3, 0, 1)],
                ),
                ("chain", [_gemm(0, 1, 0), _gemm(1, 1, 0)]),
                ("chain", [_chain_name((0, 1, 0), (1, 1, 0)), _gemm(2, 1, 0)]),
                (
                    "chain",
                    [_chain_name((0, 1, 0), (1, 1, 0), (2, 1, 0)), _gemm(3, 1, 0)],
                ),
                ("chain", [_gemm(0, 1, 1), _gemm(1, 1, 1)]),
                ("chain", [_chain_name((0, 1, 1), (1, 1, 1)), _gemm(2, 1, 1)]),
                (
                    "chain",
                    [_chain_name((0, 1, 1), (1, 1, 1), (2, 1, 1)), _gemm(3, 1, 1)],
                ),
            ],
        )
        mod(A, B, C)
        np.testing.assert_allclose(C, A @ B, atol=1e-5)
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")

    mod = df.build(
        top,
        target="aie",
        mapping_primitives=[
            ("chain", [_gemm(0, 0, 0), _gemm(1, 0, 0)]),
            ("chain", [_gemm(2, 0, 0), _gemm(3, 0, 0)]),
            ("chain", [_gemm(0, 0, 1), _gemm(1, 0, 1)]),
            ("chain", [_gemm(2, 0, 1), _gemm(3, 0, 1)]),
            ("chain", [_gemm(0, 1, 0), _gemm(1, 1, 0)]),
            ("chain", [_gemm(2, 1, 0), _gemm(3, 1, 0)]),
            ("chain", [_gemm(0, 1, 1), _gemm(1, 1, 1)]),
            ("chain", [_gemm(2, 1, 1), _gemm(3, 1, 1)]),
            (
                "bundle",
                [
                    (
                        _chain_name((0, 0, 0), (1, 0, 0)),
                        _chain_name((2, 0, 0), (3, 0, 0)),
                    ),
                    (
                        _chain_name((0, 0, 1), (1, 0, 1)),
                        _chain_name((2, 0, 1), (3, 0, 1)),
                    ),
                    (
                        _chain_name((0, 1, 0), (1, 1, 0)),
                        _chain_name((2, 1, 0), (3, 1, 0)),
                    ),
                    (
                        _chain_name((0, 1, 1), (1, 1, 1)),
                        _chain_name((2, 1, 1), (3, 1, 1)),
                    ),
                ],
            ),
        ],
    )
    mod(A, B, C)
    np.testing.assert_allclose(C, A @ B, atol=1e-5)
    print("PASSED!")


def test_pingpong_gemm_4x4x4():

    Ty = int16
    M, N, K = 128, 128, 128
    Pm, Pn, Pk = 4, 4, 4
    Mt, Nt, Kt = M // Pm, N // Pn, K // Pk

    LyA = [S(1), S(0)]
    LyB = [S(0), S(2)]
    LyC = [S(1), S(2)]

    @df.region()
    def top(A: Ty[M, K], B: Ty[K, N], C: Ty[M, N]):
        pipe: Stream[Ty[Mt, Nt], 2][Pk - 1, Pm, Pn]

        @df.kernel(mapping=[Pk, Pm, Pn], args=[A, B, C])
        def gemm(
            local_A: Ty[M, K] @ LyA, local_B: Ty[K, N] @ LyB, local_C: Ty[M, N] @ LyC
        ):
            pk, pm, pn = df.get_pid()
            C_in: Ty[Mt, Nt]
            with allo.meta_if(pk > 0):
                C_in[:, :] = pipe[pk - 1, pm, pn].get()
            with allo.meta_else():
                C_in[:, :] = 0
            C_out: Ty[Mt, Nt] = allo.add(allo.matmul(local_A, local_B), C_in)
            with allo.meta_if(pk < Pk - 1):
                pipe[pk, pm, pn].put(C_out)
            with allo.meta_elif(pk == Pk - 1):
                local_C[:, :] = C_out

    if is_available():
        mod = df.build(
            top,
            target="aie",
            mapping_primitives=[
                ("chain", [_gemm(0, 0, 0), _gemm(1, 0, 0)]),
                ("chain", [_chain_name((0, 0, 0), (1, 0, 0)), _gemm(2, 0, 0)]),
                (
                    "chain",
                    [_chain_name((0, 0, 0), (1, 0, 0), (2, 0, 0)), _gemm(3, 0, 0)],
                ),
                ("chain", [_gemm(0, 0, 1), _gemm(1, 0, 1)]),
                ("chain", [_chain_name((0, 0, 1), (1, 0, 1)), _gemm(2, 0, 1)]),
                (
                    "chain",
                    [_chain_name((0, 0, 1), (1, 0, 1), (2, 0, 1)), _gemm(3, 0, 1)],
                ),
                ("chain", [_gemm(0, 0, 2), _gemm(1, 0, 2)]),
                ("chain", [_chain_name((0, 0, 2), (1, 0, 2)), _gemm(2, 0, 2)]),
                (
                    "chain",
                    [_chain_name((0, 0, 2), (1, 0, 2), (2, 0, 2)), _gemm(3, 0, 2)],
                ),
                ("chain", [_gemm(0, 0, 3), _gemm(1, 0, 3)]),
                ("chain", [_chain_name((0, 0, 3), (1, 0, 3)), _gemm(2, 0, 3)]),
                (
                    "chain",
                    [_chain_name((0, 0, 3), (1, 0, 3), (2, 0, 3)), _gemm(3, 0, 3)],
                ),
                ("chain", [_gemm(0, 1, 0), _gemm(1, 1, 0)]),
                ("chain", [_chain_name((0, 1, 0), (1, 1, 0)), _gemm(2, 1, 0)]),
                (
                    "chain",
                    [_chain_name((0, 1, 0), (1, 1, 0), (2, 1, 0)), _gemm(3, 1, 0)],
                ),
                ("chain", [_gemm(0, 1, 1), _gemm(1, 1, 1)]),
                ("chain", [_chain_name((0, 1, 1), (1, 1, 1)), _gemm(2, 1, 1)]),
                (
                    "chain",
                    [_chain_name((0, 1, 1), (1, 1, 1), (2, 1, 1)), _gemm(3, 1, 1)],
                ),
                ("chain", [_gemm(0, 1, 2), _gemm(1, 1, 2)]),
                ("chain", [_chain_name((0, 1, 2), (1, 1, 2)), _gemm(2, 1, 2)]),
                (
                    "chain",
                    [_chain_name((0, 1, 2), (1, 1, 2), (2, 1, 2)), _gemm(3, 1, 2)],
                ),
                ("chain", [_gemm(0, 1, 3), _gemm(1, 1, 3)]),
                ("chain", [_chain_name((0, 1, 3), (1, 1, 3)), _gemm(2, 1, 3)]),
                (
                    "chain",
                    [_chain_name((0, 1, 3), (1, 1, 3), (2, 1, 3)), _gemm(3, 1, 3)],
                ),
                ("chain", [_gemm(0, 2, 0), _gemm(1, 2, 0)]),
                ("chain", [_chain_name((0, 2, 0), (1, 2, 0)), _gemm(2, 2, 0)]),
                (
                    "chain",
                    [_chain_name((0, 2, 0), (1, 2, 0), (2, 2, 0)), _gemm(3, 2, 0)],
                ),
                ("chain", [_gemm(0, 2, 1), _gemm(1, 2, 1)]),
                ("chain", [_chain_name((0, 2, 1), (1, 2, 1)), _gemm(2, 2, 1)]),
                (
                    "chain",
                    [_chain_name((0, 2, 1), (1, 2, 1), (2, 2, 1)), _gemm(3, 2, 1)],
                ),
                ("chain", [_gemm(0, 2, 2), _gemm(1, 2, 2)]),
                ("chain", [_chain_name((0, 2, 2), (1, 2, 2)), _gemm(2, 2, 2)]),
                (
                    "chain",
                    [_chain_name((0, 2, 2), (1, 2, 2), (2, 2, 2)), _gemm(3, 2, 2)],
                ),
                ("chain", [_gemm(0, 2, 3), _gemm(1, 2, 3)]),
                ("chain", [_chain_name((0, 2, 3), (1, 2, 3)), _gemm(2, 2, 3)]),
                (
                    "chain",
                    [_chain_name((0, 2, 3), (1, 2, 3), (2, 2, 3)), _gemm(3, 2, 3)],
                ),
                ("chain", [_gemm(0, 3, 0), _gemm(1, 3, 0)]),
                ("chain", [_chain_name((0, 3, 0), (1, 3, 0)), _gemm(2, 3, 0)]),
                (
                    "chain",
                    [_chain_name((0, 3, 0), (1, 3, 0), (2, 3, 0)), _gemm(3, 3, 0)],
                ),
                ("chain", [_gemm(0, 3, 1), _gemm(1, 3, 1)]),
                ("chain", [_chain_name((0, 3, 1), (1, 3, 1)), _gemm(2, 3, 1)]),
                (
                    "chain",
                    [_chain_name((0, 3, 1), (1, 3, 1), (2, 3, 1)), _gemm(3, 3, 1)],
                ),
                ("chain", [_gemm(0, 3, 2), _gemm(1, 3, 2)]),
                ("chain", [_chain_name((0, 3, 2), (1, 3, 2)), _gemm(2, 3, 2)]),
                (
                    "chain",
                    [_chain_name((0, 3, 2), (1, 3, 2), (2, 3, 2)), _gemm(3, 3, 2)],
                ),
                ("chain", [_gemm(0, 3, 3), _gemm(1, 3, 3)]),
                ("chain", [_chain_name((0, 3, 3), (1, 3, 3)), _gemm(2, 3, 3)]),
                (
                    "chain",
                    [_chain_name((0, 3, 3), (1, 3, 3), (2, 3, 3)), _gemm(3, 3, 3)],
                ),
            ],
        )
        A = np.random.randint(0, 64, (M, K)).astype(np.int16)
        B = np.random.randint(0, 64, (K, N)).astype(np.int16)
        C = np.zeros((M, N)).astype(np.int16)
        mod(A, B, C)
        np.testing.assert_allclose(C, A @ B, atol=1e-5)
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


def test_split_k_gemm_1x1x4():

    Ty = int16
    M, N, K = 32, 32, 128
    Pk = 4

    LyA = [R, S(0)]
    LyB = [S(0), R]

    @df.region()
    def top(A: Ty[M, K], B: Ty[K, N], C: Ty[M, N]):
        pipe: Stream[Ty[M, N], 2][Pk]

        @df.kernel(mapping=[Pk], args=[A, B])
        def partial_gemm(local_A: Ty[M, K] @ LyA, local_B: Ty[K, N] @ LyB):
            pk = df.get_pid()
            pipe[pk].put(allo.matmul(local_A, local_B))

        @df.kernel(mapping=[1], args=[C])
        def acc(local_C: Ty[M, N]):
            C_: Ty[M, N] = 0
            with allo.meta_for(Pk) as i:
                C_[:, :] += pipe[i].get()
            local_C[:, :] = C_

    if is_available():
        mod = df.build(
            top,
            target="aie",
            mapping_primitives=[
                (
                    "bundle",
                    [
                        _partial_gemm(0),
                        _partial_gemm(1),
                        _partial_gemm(2),
                        _partial_gemm(3),
                    ],
                )
            ],
        )
        A = np.random.randint(0, 64, (M, K)).astype(np.int16)
        B = np.random.randint(0, 64, (K, N)).astype(np.int16)
        C = np.zeros((M, N)).astype(np.int16)
        mod(A, B, C)
        np.testing.assert_allclose(C, A @ B, atol=1e-5)
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


def test_split_k_gemm_2x2x4():

    Ty = int16
    M, N, K = 64, 64, 128
    Pm, Pn = 2, 2
    Pk = 4
    Mt, Nt = M // Pm, N // Pn

    LyA = [S(1), S(0)]
    LyB = [S(0), S(2)]
    LyC = [S(1), S(2)]

    @df.region()
    def top(A: Ty[M, K], B: Ty[K, N], C: Ty[M, N]):
        pipe: Stream[Ty[Mt, Nt], 2][Pk, Pm, Pn]

        @df.kernel(mapping=[Pk, Pm, Pn], args=[A, B])
        def partial_gemm(local_A: Ty[M, K] @ LyA, local_B: Ty[K, N] @ LyB):
            pk, pm, pn = df.get_pid()
            pipe[pk, pm, pn].put(allo.matmul(local_A, local_B))

        @df.kernel(mapping=[1, Pm, Pn], args=[C])
        def acc(local_C: Ty[M, N] @ LyC):
            _, pm, pn = df.get_pid()
            C_: Ty[Mt, Nt] = 0
            with allo.meta_for(Pk) as i:
                C_[:, :] += pipe[i, pm, pn].get()
            local_C[:, :] = C_

    if is_available():
        mod = df.build(
            top,
            target="aie",
            mapping_primitives=[
                (
                    "bundle",
                    [
                        _partial_gemm(0, 0, 0),
                        _partial_gemm(1, 0, 0),
                        _partial_gemm(2, 0, 0),
                        _partial_gemm(3, 0, 0),
                    ],
                ),
                (
                    "bundle",
                    [
                        _partial_gemm(0, 0, 1),
                        _partial_gemm(1, 0, 1),
                        _partial_gemm(2, 0, 1),
                        _partial_gemm(3, 0, 1),
                    ],
                ),
                (
                    "bundle",
                    [
                        _partial_gemm(0, 1, 0),
                        _partial_gemm(1, 1, 0),
                        _partial_gemm(2, 1, 0),
                        _partial_gemm(3, 1, 0),
                    ],
                ),
                (
                    "bundle",
                    [
                        _partial_gemm(0, 1, 1),
                        _partial_gemm(1, 1, 1),
                        _partial_gemm(2, 1, 1),
                        _partial_gemm(3, 1, 1),
                    ],
                ),
            ],
        )
        A = np.random.randint(0, 64, (M, K)).astype(np.int16)
        B = np.random.randint(0, 64, (K, N)).astype(np.int16)
        C = np.zeros((M, N)).astype(np.int16)
        mod(A, B, C)
        np.testing.assert_allclose(C, A @ B, atol=1e-5)
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


if __name__ == "__main__":
    test_gemm_2D_v1()
    test_gemm_2D_v2()
    test_pingpong_gemm_2x2x2()
    test_pingpong_gemm_2x2x2_partial_chain()
    test_pingpong_gemm_1x1x4()
    test_pingpong_gemm_2x2x4()
    test_pingpong_gemm_4x4x4()
    test_split_k_gemm_1x1x4()
    test_split_k_gemm_2x2x4()
