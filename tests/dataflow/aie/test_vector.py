# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import allo
from allo.ir.types import int32, float32, bfloat16
import allo.dataflow as df
import numpy as np
from allo.memory import Layout

Ly = Layout("S0")


def _test_vector_scalar_add():
    # https://github.com/Xilinx/mlir-aie/tree/main/programming_examples/basic/vector_scalar_add
    Ty = int32
    M = 1024

    @df.region()
    def top():
        @df.kernel(mapping=[1])
        def core(A: Ty[M], B: Ty[M]):
            B[:] = allo.add(A, 1)

    A = np.random.randint(0, 100, M).astype(np.int32)
    if "MLIR_AIE_INSTALL_DIR" in os.environ:
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


def _test_vector_scalar_mul():
    # https://github.com/Xilinx/mlir-aie/tree/main/programming_examples/basic/vector_scalar_mul
    Ty = float32
    M = 512

    @df.region()
    def top():
        @df.kernel(mapping=[1])
        def core(A: Ty[M], B: Ty[M]):
            B[:] = allo.mul(A, 2)

    A = np.random.random(M).astype(np.float32)
    if "MLIR_AIE_INSTALL_DIR" in os.environ:
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


def _test_vector_vector_add():
    # # https://github.com/Xilinx/mlir-aie/tree/main/programming_examples/basic/vector_vector_add
    Ty = int32
    M = 1024

    @df.region()
    def top():
        @df.kernel(mapping=[1])
        def core(A: Ty[M], B: Ty[M], C: Ty[M]):
            C[:] = allo.add(A, B)

    A = np.random.randint(0, 100, M).astype(np.int32)
    B = np.random.randint(0, 100, M).astype(np.int32)
    if "MLIR_AIE_INSTALL_DIR" in os.environ:
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


def _test_vector_vector_bf16_add():
    from ml_dtypes import bfloat16 as np_bfloat16

    Ty = bfloat16
    M = 1024

    @df.region()
    def top():
        @df.kernel(mapping=[1])
        def core(A: Ty[M], B: Ty[M], C: Ty[M]):
            C[:] = allo.add(A, B)

    A = np.random.random(M).astype(np_bfloat16)
    B = np.random.random(M).astype(np_bfloat16)
    if "MLIR_AIE_INSTALL_DIR" in os.environ:
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


def _test_vector_vector_mul():
    # https://github.com/Xilinx/mlir-aie/tree/main/programming_examples/basic/vector_vector_mul
    Ty = float32
    M = 1024

    @df.region()
    def top():
        @df.kernel(mapping=[1])
        def core(A: Ty[M], B: Ty[M], C: Ty[M]):
            C[:] = allo.mul(A, B)

    A = np.random.random(M).astype(np.float32)
    B = np.random.random(M).astype(np.float32)

    if "MLIR_AIE_INSTALL_DIR" in os.environ:
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


def _test_vector_scalar_add_p0():
    # https://github.com/Xilinx/mlir-aie/tree/main/programming_guide/section-2/section-2d
    #                |--------------------------------------------|
    #                v   v-------------------------v              v
    # shim tile <-> mem tile <-> comp tile0    comp tile1    comp tile2
    Ty = int32
    M = 1024
    P0 = 4

    @df.region()
    def top():
        @df.kernel(mapping=[P0])
        def core(A: Ty[M] @ Ly, B: Ty[M] @ Ly):
            B[:] = allo.add(A[:], 1)

    mod = df.build(top, target="aie")
    A = np.random.randint(0, 100, M).astype(np.int32)
    B = np.zeros(M).astype(np.int32)
    mod(A, B)
    np.testing.assert_allclose(B, A + 1)
    print("PASSED!")


def _test_vector_vector_add_p0():
    #                  |--------------------------------------------|
    #                  v   v--------------------------v             v
    # shim tile <-> A mem tile 0 <-> comp tile0    comp tile1    comp tile2
    #       ^-----> B mem tile 1 <-------^------------^-------------^
    Ty = int32
    M = 1024
    P0 = 4

    @df.region()
    def top():
        @df.kernel(mapping=[P0])
        def core(A: Ty[M] @ Ly, B: Ty[M] @ Ly, C: Ty[M] @ Ly):
            C[:] = allo.add(A, B)

    mod = df.build(top, target="aie")
    A = np.random.randint(0, 100, M).astype(np.int32)
    B = np.random.randint(0, 100, M).astype(np.int32)
    C = np.zeros(M).astype(np.int32)
    mod(A, B, C)
    np.testing.assert_allclose(C, A + B)
    print("PASSED!")


if __name__ == "__main__":
    _test_vector_scalar_add()
    _test_vector_scalar_mul()
    _test_vector_vector_add()
    _test_vector_vector_bf16_add()
    _test_vector_vector_mul()
    _test_vector_scalar_add_p0()
    _test_vector_vector_add_p0()
