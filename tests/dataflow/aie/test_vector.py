# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import int32, float32
import allo.dataflow as df
import numpy as np


def _test_vector_scalar_add():
    # https://github.com/Xilinx/mlir-aie/tree/main/programming_examples/basic/vector_scalar_add
    Ty = int32
    M = 1024

    @df.region()
    def top():
        @df.kernel(mapping=[1])
        def core(A: Ty[M], B: Ty[M]):
            B[:] = allo.add(A, 1)

    mod = df.build(top, target="aie")
    A = np.random.randint(0, 100, M).astype(np.int32)
    B = np.zeros(M).astype(np.int32)
    mod(A, B)
    np.testing.assert_allclose(B, A + 1)
    print("PASSED!")


def _test_vector_scalar_mul():
    # https://github.com/Xilinx/mlir-aie/tree/main/programming_examples/basic/vector_scalar_mul
    Ty = float32
    M = 512

    @df.region()
    def top():
        @df.kernel(mapping=[1])
        def core(A: Ty[M], B: Ty[M]):
            B[:] = allo.mul(A, 2)

    mod = df.build(top, target="aie")
    A = np.random.random(M).astype(np.float32)
    B = np.zeros(M).astype(np.float32)
    mod(A, B)
    np.testing.assert_allclose(B, A * 2, rtol=1e-5)
    print("PASSED!")


def _test_vector_vector_add():
    # # https://github.com/Xilinx/mlir-aie/tree/main/programming_examples/basic/vector_vector_add
    Ty = int32
    M = 1024

    @df.region()
    def top():
        @df.kernel(mapping=[1])
        def core(A: Ty[M], B: Ty[M], C: Ty[M]):
            C[:] = allo.add(A, B)

    mod = df.build(top, target="aie")
    A = np.random.randint(0, 100, M).astype(np.int32)
    B = np.random.randint(0, 100, M).astype(np.int32)
    C = np.zeros(M).astype(np.int32)
    mod(A, B, C)
    np.testing.assert_allclose(C, A + B)
    print("PASSED!")


def _test_vector_vector_mul():
    # https://github.com/Xilinx/mlir-aie/tree/main/programming_examples/basic/vector_vector_mul
    Ty = float32
    M = 1024

    @df.region()
    def top():
        @df.kernel(mapping=[1])
        def core(A: Ty[M], B: Ty[M], C: Ty[M]):
            C[:] = allo.mul(A, B)

    mod = df.build(top, target="aie")
    A = np.random.random(M).astype(np.float32)
    B = np.random.random(M).astype(np.float32)
    C = np.zeros(M).astype(np.float32)
    mod(A, B, C)
    np.testing.assert_allclose(C, A * B, rtol=1e-5)
    print("PASSED!")


if __name__ == "__main__":
    _test_vector_scalar_add()
    _test_vector_scalar_mul()
    _test_vector_vector_add()
    _test_vector_vector_mul()
