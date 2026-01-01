# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import os

import pytest
import allo
from allo.ir.types import int4, int8, Stream
import allo.dataflow as df
from allo.memory import Layout
from allo.backend.aie import is_available


@pytest.fixture(scope="module", autouse=True)
def setup_env():
    dir_path = os.path.dirname(os.path.abspath(__file__))
    os.environ["ALLO_EXTERNAL_KERNEL_DIR"] = (
        f"{dir_path}/../../../allo/library/aie/kernels/"
    )
    yield
    del os.environ["ALLO_EXTERNAL_KERNEL_DIR"]


@pytest.mark.skipif(
    os.environ.get("NPU2") == "1",
    reason="Skipped because external kernel not supported on XDNA2",
)
def test_gemm_1D():
    TyI = int4
    TyO = int8
    M, N, K = 64, 64, 64
    P0 = 2

    LyA = Layout("S0R")

    @df.region()
    def top(A: TyI[M, K], B: TyI[K, N], C: TyO[M, N]):
        @df.kernel(mapping=[P0], args=[A, B, C])
        def gemm(
            local_A: TyI[M, K] @ LyA, local_B: TyI[K, N], local_C: TyO[M, N] @ LyA
        ):
            local_C[:, :] = allo.matmul(local_A, local_B)

    if is_available():
        mod = df.build(top, target="aie")
        A = np.random.randint(-2, 2, (M, K)).astype(np.int8)
        B = np.random.randint(-2, 2, (K, N)).astype(np.int8)
        C = np.zeros((M, N)).astype(np.int8)
        mod(A, B, C)
        np.testing.assert_allclose(C, A @ B, atol=1e-5)
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


@pytest.mark.skipif(
    os.environ.get("NPU2") == "1",
    reason="Skipped because external kernel not supported on XDNA2",
)
def test_gemm_2D():
    TyI = int4
    TyO = int8
    M, N, K = 64, 64, 64
    P0, P1 = 2, 2
    LyA = Layout("S0R")
    LyB = Layout("RS1")
    LyC = Layout("S0S1")

    @df.region()
    def top(A: TyI[M, K], B: TyI[K, N], C: TyO[M, N]):
        @df.kernel(mapping=[P0, P1], args=[A, B, C])
        def gemm(
            local_A: TyI[M, K] @ LyA, local_B: TyI[K, N] @ LyB, local_C: TyO[M, N] @ LyC
        ):
            local_C[:, :] = allo.matmul(local_A, local_B)

    if is_available():
        mod = df.build(top, target="aie")
        A = np.random.randint(-4, 4, (M, K)).astype(np.int8)
        B = np.random.randint(-4, 4, (K, N)).astype(np.int8)
        C = np.zeros((M, N)).astype(np.int8)
        mod(A, B, C)
        np_C = A.astype(np.int8) @ B.astype(np.int8)
        np.testing.assert_allclose(C, np_C, atol=1e-5)
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


@pytest.mark.skipif(
    os.environ.get("NPU2") == "1",
    reason="Skipped because external kernel not supported on XDNA2",
)
def test_mixed_gemm_1D():
    Ty = int8
    Ty_l = int4
    M, N, K = 64, 64, 64
    P0 = 2

    LyA = Layout("S0R")

    @df.region()
    def top(A: Ty_l[M, K], B: Ty_l[K, N], C: Ty[M, N]):
        @df.kernel(mapping=[P0], args=[A, B, C])
        def gemm(
            local_A: Ty_l[M, K] @ LyA, local_B: Ty_l[K, N], local_C: Ty[M, N] @ LyA
        ):
            local_C[:, :] = allo.matmul(local_A, local_B)

    if is_available():
        mod = df.build(top, target="aie")
        A = np.random.randint(-2, 2, (M, K)).astype(np.int8)
        B = np.random.randint(-2, 2, (K, N)).astype(np.int8)
        C = np.zeros((M, N)).astype(np.int8)
        mod(A, B, C)
        np.testing.assert_allclose(C, A @ B, atol=1e-5)
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


@pytest.mark.skipif(
    os.environ.get("NPU2") == "1",
    reason="Skipped because external kernel not supported on XDNA2",
)
def test_mixed_gemm_2D():
    Ty = int8
    Ty_l = int4
    M, N, K = 64, 64, 64
    P0, P1 = 2, 2
    LyA = Layout("S0R")
    LyB = Layout("RS1")
    LyC = Layout("S0S1")

    @df.region()
    def top(A: Ty[M, K], B: Ty_l[K, N], C: Ty[M, N]):
        @df.kernel(mapping=[P0, P1], args=[A, B, C])
        def gemm(
            local_A: Ty[M, K] @ LyA, local_B: Ty_l[K, N] @ LyB, local_C: Ty[M, N] @ LyC
        ):
            local_C[:, :] = allo.matmul(local_A, local_B)

    if is_available():
        mod = df.build(top, target="aie")
        A = np.random.randint(-4, 4, (M, K)).astype(np.int8)
        B = np.random.randint(-4, 4, (K, N)).astype(np.int8)
        C = np.zeros((M, N)).astype(np.int8)
        mod(A, B, C)
        np_C = A.astype(np.int8) @ B.astype(np.int8)
        np.testing.assert_allclose(C, np_C, atol=1e-5)
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


def gen_gemm_mapping_primitive(Pm, Pn, Pk, col_num=4, row_num=4):
    # chain on k dimension
    mapping_primitives = []
    bases: list[list[str]] = []
    for i in range(Pm):
        bases.append([])
        for j in range(Pn):
            base = f"gemm_0_{i}_{j}"
            for k in range(1, Pk):
                mapping_primitives.append(("chain", [base, f"gemm_{k}_{i}_{j}"]))
                base += f"-gemm_{k}_{i}_{j}"
            bases[i].append(base)

    if Pn // col_num < 1 or Pm // row_num < 1:
        col_num, row_num = row_num, col_num
    if Pn < col_num:
        col_num = Pn
    if Pm < row_num:
        row_num = Pm
    if Pn // col_num > 1 or Pm // row_num > 1:
        for i in range(row_num):
            for j in range(col_num):
                bundle_list = []
                for p in range(Pm // row_num):
                    for q in range(Pn // col_num):
                        bundle_list.append(bases[i + row_num * p][j + col_num * q])
                mapping_primitives.append(("bundle", bundle_list))

    return mapping_primitives


@pytest.mark.skipif(
    os.environ.get("NPU2") == "1",
    reason="Skipped because external kernel not supported on XDNA2",
)
@pytest.mark.parametrize(
    "M, N, K, Pm, Pn, Pk",
    [
        (512, 512, 512, 8, 4, 4),
    ],
)
def test_pingpong_mixed_gemm(M, N, K, Pm, Pn, Pk):
    TyI = int8
    TyI_l = int4
    TyO = int8
    Mt, Nt = M // Pm, N // Pn

    LyA = Layout("S1S2")
    LyB = Layout("S2S0")
    LyC = Layout("S1S0")

    @df.region()
    def top(A: TyI[M, K], B: TyI_l[K, N], C: TyO[M, N]):
        pipe: Stream[TyO[Mt, Nt], 2][Pk - 1, Pm, Pn]

        @df.kernel(mapping=[Pk, Pm, Pn], args=[A, B, C])
        def gemm(
            local_A: TyI[M, K] @ LyA,
            local_B: TyI_l[K, N] @ LyB,
            local_C: TyO[M, N] @ LyC,
        ):
            pk, pm, pn = df.get_pid()
            C_in: TyO[Mt, Nt]
            with allo.meta_if(pk > 0):
                C_in[:, :] = pipe[pk - 1, pm, pn].get()
            with allo.meta_else():
                C_in[:, :] = 0
            C_out: TyO[Mt, Nt] = allo.add(allo.matmul(local_A, local_B), C_in)
            with allo.meta_if(pk < Pk - 1):
                pipe[pk, pm, pn].put(C_out)
            with allo.meta_elif(pk == Pk - 1):
                local_C[:, :] = C_out

    mapping_primitives = gen_gemm_mapping_primitive(Pm, Pn, Pk)

    if is_available():
        mod = df.build(
            top,
            target="aie",
            mapping_primitives=mapping_primitives,
            profile=False,
            warmup=200,
            num_iters=1000,
        )
        A = np.random.randint(-2, 2, (M, K)).astype(np.int8)
        B = np.random.randint(-2, 2, (K, N)).astype(np.int8)
        C = np.zeros((M, N)).astype(np.int8)
        mod(A, B, C)
        np.testing.assert_allclose(C, A @ B, atol=1e-5)
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


if __name__ == "__main__":
    # [NOTE]: export ALLO_EXTERNAL_KERNEL_DIR=/allo/root/dir/allo/library/aie/kernels/
    dir_path = os.path.dirname(os.path.abspath(__file__))
    os.environ["ALLO_EXTERNAL_KERNEL_DIR"] = (
        f"{dir_path}/../../../allo/library/aie/kernels/"
    )

    test_gemm_1D()
    test_gemm_2D()
    test_mixed_gemm_1D()
    test_mixed_gemm_2D()
    test_pingpong_mixed_gemm(512, 512, 512, 8, 4, 4)

    del os.environ["ALLO_EXTERNAL_KERNEL_DIR"]
