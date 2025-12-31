# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import allo
from allo.ir.types import int32, float32, bfloat16
import allo.dataflow as df
import numpy as np
from allo.memory import Layout
from allo.backend.aie import is_available

Ly = Layout("S0")


def test_vector_scalar_add():
    # https://github.com/Xilinx/mlir-aie/tree/main/programming_examples/basic/vector_scalar_add
    Ty = int32
    M = 1024

    @df.region()
    def top(A: Ty[M], B: Ty[M]):
        @df.kernel(mapping=[1], args=[A, B])
        def core(local_A: Ty[M], local_B: Ty[M]):
            local_B[:] = allo.add(local_A, 1)

    A = np.random.randint(0, 100, M).astype(np.int32)
    if is_available():
        mod = df.build(top, target="aie")
        B = np.zeros(M).astype(np.int32)
        mod(A, B)
        np.testing.assert_allclose(B, A + 1)
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")
    sim_mod = df.build(top, target="simulator")
    B = np.zeros(M).astype(np.int32)
    sim_mod(A, B)
    np.testing.assert_allclose(B, A + 1)
    print("Dataflow Simulator Passed!")


def test_vector_scalar_conditional_add():
    Ty = int32
    M = 1024
    P = 4

    @df.region()
    def top(A: Ty[M], B: Ty[M]):
        @df.kernel(mapping=[P], args=[A, B])
        def core(local_A: Ty[M] @ Ly, local_B: Ty[M] @ Ly):
            pi = df.get_pid()
            with allo.meta_if(pi < P // 2):
                local_B[:] = allo.add(local_A, 1)
            with allo.meta_else():
                local_B[:] = allo.add(local_A, -1)

    A = np.random.randint(0, 100, M).astype(np.int32)
    if is_available():
        mod = df.build(top, target="aie")
        B = np.zeros(M).astype(np.int32)
        mod(A, B)
        A[: M // 2] += 1
        A[M // 2 :] -= 1
        np.testing.assert_allclose(B, A)
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


def test_vector_scalar_mul():
    # https://github.com/Xilinx/mlir-aie/tree/main/programming_examples/basic/vector_scalar_mul
    Ty = float32
    M = 512

    @df.region()
    def top(A: Ty[M], B: Ty[M]):
        @df.kernel(mapping=[1], args=[A, B])
        def core(local_A: Ty[M], local_B: Ty[M]):
            local_B[:] = allo.mul(local_A, 2)

    A = np.random.random(M).astype(np.float32)
    if is_available():
        mod = df.build(top, target="aie")
        B = np.zeros(M).astype(np.float32)
        mod(A, B)
        np.testing.assert_allclose(B, A * 2, rtol=1e-5)
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")
    sim_mod = df.build(top, target="simulator")
    B = np.zeros(M).astype(np.float32)
    sim_mod(A, B)
    np.testing.assert_allclose(B, A * 2, rtol=1e-5)
    print("Dataflow Simulator Passed!")


def test_vector_vector_add():
    # # https://github.com/Xilinx/mlir-aie/tree/main/programming_examples/basic/vector_vector_add
    Ty = int32
    M = 1024

    @df.region()
    def top(A: Ty[M], B: Ty[M], C: Ty[M]):
        @df.kernel(mapping=[1], args=[A, B, C])
        def core(local_A: Ty[M], local_B: Ty[M], local_C: Ty[M]):
            local_C[:] = allo.add(local_A, local_B)

    A = np.random.randint(0, 100, M).astype(np.int32)
    B = np.random.randint(0, 100, M).astype(np.int32)
    if is_available():
        mod = df.build(top, target="aie")
        C = np.zeros(M).astype(np.int32)
        mod(A, B, C)
        np.testing.assert_allclose(C, A + B)
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")
    sim_mod = df.build(top, target="simulator")
    C = np.zeros(M).astype(np.int32)
    sim_mod(A, B, C)
    np.testing.assert_allclose(C, A + B)
    print("Dataflow Simulator Passed!")


def test_vector_vector_bf16_add():
    from ml_dtypes import bfloat16 as np_bfloat16

    Ty = bfloat16
    M = 1024

    @df.region()
    def top(A: Ty[M], B: Ty[M], C: Ty[M]):
        @df.kernel(mapping=[1], args=[A, B, C])
        def core(local_A: Ty[M], local_B: Ty[M], local_C: Ty[M]):
            local_C[:] = allo.add(local_A, local_B)

    A = np.random.random(M).astype(np_bfloat16)
    B = np.random.random(M).astype(np_bfloat16)
    if is_available():
        mod = df.build(top, target="aie")
        C = np.zeros(M).astype(np_bfloat16)
        mod(A, B, C)
        np.testing.assert_allclose(
            C.astype(np.float32), (A + B).astype(np.float32), rtol=1e-2
        )
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")
    sim_mod = df.build(top, target="simulator")
    C = np.zeros(M).astype(np_bfloat16)
    sim_mod(A, B, C)
    np.testing.assert_allclose(
        C.astype(np.float32), (A + B).astype(np.float32), rtol=1e-2
    )
    print("Dataflow Simulator Passed!")


def test_vector_vector_mul():
    # https://github.com/Xilinx/mlir-aie/tree/main/programming_examples/basic/vector_vector_mul
    Ty = float32
    M = 1024

    @df.region()
    def top(A: Ty[M], B: Ty[M], C: Ty[M]):
        @df.kernel(mapping=[1], args=[A, B, C])
        def core(local_A: Ty[M], local_B: Ty[M], local_C: Ty[M]):
            local_C[:] = allo.mul(local_A, local_B)

    A = np.random.random(M).astype(np.float32)
    B = np.random.random(M).astype(np.float32)

    if is_available():
        mod = df.build(top, target="aie")
        C = np.zeros(M).astype(np.float32)
        mod(A, B, C)
        np.testing.assert_allclose(C, A * B, rtol=1e-5)
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")

    sim_mod = df.build(top, target="simulator")
    C = np.zeros(M).astype(np.float32)
    sim_mod(A, B, C)
    np.testing.assert_allclose(C, A * B, atol=1e-5)
    print("Dataflow Simulator Passed!")


def test_vector_scalar_add_p0():
    # https://github.com/Xilinx/mlir-aie/tree/main/programming_guide/section-2/section-2d
    #                |--------------------------------------------|
    #                v   v-------------------------v              v
    # shim tile <-> mem tile <-> comp tile0    comp tile1    comp tile2
    Ty = int32
    M = 1024
    P0 = 4

    @df.region()
    def top(A: Ty[M], B: Ty[M]):
        @df.kernel(mapping=[P0], args=[A, B])
        def core(local_A: Ty[M], local_B: Ty[M]):
            local_B[:] = allo.add(local_A[:], 1)

    if is_available():
        mod = df.build(top, target="aie")
        A = np.random.randint(0, 100, M).astype(np.int32)
        B = np.zeros(M).astype(np.int32)
        mod(A, B)
        np.testing.assert_allclose(B, A + 1)
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


def test_vector_vector_add_p0():
    #                  |--------------------------------------------|
    #                  v   v--------------------------v             v
    # shim tile <-> A mem tile 0 <-> comp tile0    comp tile1    comp tile2
    #       ^-----> B mem tile 1 <-------^------------^-------------^
    Ty = int32
    M = 1024
    P0 = 4

    @df.region()
    def top(A: Ty[M], B: Ty[M], C: Ty[M]):
        @df.kernel(mapping=[P0], args=[A, B, C])
        def core(local_A: Ty[M], local_B: Ty[M], local_C: Ty[M]):
            local_C[:] = allo.add(local_A, local_B)

    if is_available():
        mod = df.build(top, target="aie")
        A = np.random.randint(0, 100, M).astype(np.int32)
        B = np.random.randint(0, 100, M).astype(np.int32)
        C = np.zeros(M).astype(np.int32)
        mod(A, B, C)
        np.testing.assert_allclose(C, A + B)
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


if __name__ == "__main__":
    test_vector_scalar_add()
    test_vector_scalar_conditional_add()
    test_vector_scalar_mul()
    test_vector_vector_add()
    test_vector_vector_bf16_add()
    test_vector_vector_mul()
    test_vector_scalar_add_p0()
    test_vector_vector_add_p0()
