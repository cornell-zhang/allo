# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
import numpy as np
from allo.ir.types import float32
import allo.dataflow as df
from allo.backend.aie.vliw import create_vliw_module
from allo.backend.aie import vliw


def test_simple_add():
    """Test simple addition function on AIE"""

    # Define simple addition function in Allo
    def simple_add(a: float32[16], b: float32[16]) -> float32[16]:
        c: float32[16] = 0.0
        for i in allo.grid(16):
            c[i] = a[i] + b[i]
        return c

    # Create AIE VLIW module
    aie_mod = create_vliw_module(simple_add)
    simple_add_ext = aie_mod.get_external_module()

    # Use in actual AIE dataflow:
    @df.region()
    def my_aie_design():
        @df.kernel(mapping=[1])
        def core(
            A: float32[16],
            B: float32[16],
            C: float32[16],
        ):
            simple_add_ext(A, B, C)

    aie_module = df.build(my_aie_design, target="aie")
    A = np.random.rand(16).astype(np.float32)
    B = np.random.rand(16).astype(np.float32)
    C = np.zeros(16).astype(np.float32)
    aie_module(A, B, C)
    # verify the result
    np.testing.assert_allclose(C, A + B)
    print("✓ AIE design verification successful")


def test_matrix_multiply():
    """Test matrix multiplication function on AIE"""

    # Define matrix multiply function in Allo
    def matrix_multiply(a: float32[4, 4], b: float32[4, 4]) -> float32[4, 4]:
        c: float32[4, 4] = 0.0
        for i in allo.grid(4):
            for j in allo.grid(4):
                for k in allo.grid(4):
                    c[i, j] += a[i, k] * b[k, j]
        return c

    # Create AIE VLIW module for matrix multiply
    matmul_aie_mod = create_vliw_module(matrix_multiply)
    matmul_ext = matmul_aie_mod.get_external_module()

    # Use matrix multiply in AIE dataflow:
    @df.region()
    def matmul_aie_design():
        @df.kernel(mapping=[1])
        def core(
            A: float32[4, 4],
            B: float32[4, 4],
            C: float32[4, 4],
        ):
            matmul_ext(A, B, C)

    matmul_aie_module = df.build(matmul_aie_design, target="aie")
    A_mat = np.random.rand(4, 4).astype(np.float32)
    B_mat = np.random.rand(4, 4).astype(np.float32)
    C_mat = np.zeros((4, 4)).astype(np.float32)
    matmul_aie_module(A_mat, B_mat, C_mat)
    # verify the result
    expected_mat = np.matmul(A_mat, B_mat)
    np.testing.assert_allclose(C_mat, expected_mat, rtol=1e-5)
    print("✓ Matrix multiply AIE design verification successful")


def test_conv2d():
    """Test 2D convolution function on AIE"""

    # Define 2D convolution function in Allo
    # Input: 4x4 image, Kernel: 3x3, Output: 2x2 (valid padding)
    def conv2d(input_img: float32[4, 4], kernel: float32[3, 3]) -> float32[2, 2]:
        output: float32[2, 2] = 0.0

        # Perform convolution with valid padding
        for i in allo.grid(2):  # Output height
            for j in allo.grid(2):  # Output width
                sum_val: float32 = 0.0
                for ki in allo.grid(3):  # Kernel height
                    for kj in allo.grid(3):  # Kernel width
                        sum_val += input_img[i + ki, j + kj] * kernel[ki, kj]
                output[i, j] = sum_val

        return output

    # Create AIE VLIW module for conv2d
    conv2d_aie_mod = create_vliw_module(conv2d)
    conv2d_ext = conv2d_aie_mod.get_external_module()

    # Use conv2d in AIE dataflow:
    @df.region()
    def conv2d_aie_design():
        @df.kernel(mapping=[1])
        def core(
            Input: float32[4, 4],
            Kernel: float32[3, 3],
            Output: float32[2, 2],
        ):
            conv2d_ext(Input, Kernel, Output)

    conv2d_aie_module = df.build(conv2d_aie_design, target="aie")
    Input_img = np.random.rand(4, 4).astype(np.float32)  # Input image
    Kernel_weights = np.random.rand(3, 3).astype(np.float32)  # Convolution kernel
    Output_conv = np.zeros((2, 2)).astype(np.float32)
    conv2d_aie_module(Input_img, Kernel_weights, Output_conv)

    # verify the result with native numpy implementation
    def conv2d_reference(input_img, kernel):
        output = np.zeros((2, 2), dtype=np.float32)
        for i in range(2):
            for j in range(2):
                output[i, j] = np.sum(input_img[i : i + 3, j : j + 3] * kernel)
        return output

    expected_conv = conv2d_reference(Input_img, Kernel_weights)
    np.testing.assert_allclose(Output_conv, expected_conv, rtol=1e-5)
    print("✓ 2D Convolution AIE design verification successful")


def test_simple_add_with_decorator():
    """Test simple addition function with @vliw.kernel decorator (new approach)"""

    # Define simple addition function with decorator
    @vliw.kernel()
    def simple_add(a: float32[16], b: float32[16]) -> float32[16]:
        c: float32[16] = 0.0
        for i in allo.grid(16):
            c[i] = a[i] + b[i]
        return c

    # Use directly in AIE dataflow - no need for create_vliw_module or get_external_module
    @df.region()
    def my_aie_design():
        @df.kernel(mapping=[1])
        def core(
            A: float32[16],
            B: float32[16],
            C: float32[16],
        ):
            simple_add(A, B, C)  # Direct call!

    aie_module = df.build(my_aie_design, target="aie")
    A = np.random.rand(16).astype(np.float32)
    B = np.random.rand(16).astype(np.float32)
    C = np.zeros(16).astype(np.float32)
    aie_module(A, B, C)
    # verify the result
    np.testing.assert_allclose(C, A + B)
    print("✓ AIE design with @vliw.kernel decorator verification successful")


def test_vliw_decorator_matrix_multiply():
    """Test the @vliw.kernel decorator with matrix multiplication"""

    # Define function with @vliw.kernel decorator and custom project path
    @vliw.kernel(project="custom_matmul_project")
    def matrix_multiply(a: float32[4, 4], b: float32[4, 4]) -> float32[4, 4]:
        c: float32[4, 4] = 0.0
        for i in allo.grid(4):
            for j in allo.grid(4):
                for k in allo.grid(4):
                    c[i, j] += a[i, k] * b[k, j]
        return c

    # Use directly in AIE dataflow
    @df.region()
    def matmul_aie_design():
        @df.kernel(mapping=[1])
        def core(
            A: float32[4, 4],
            B: float32[4, 4],
            C: float32[4, 4],
        ):
            matrix_multiply(A, B, C)

    matmul_aie_module = df.build(matmul_aie_design, target="aie")
    A_mat = np.random.rand(4, 4).astype(np.float32)
    B_mat = np.random.rand(4, 4).astype(np.float32)
    C_mat = np.zeros((4, 4)).astype(np.float32)
    matmul_aie_module(A_mat, B_mat, C_mat)
    # verify the result
    expected_mat = np.matmul(A_mat, B_mat)
    np.testing.assert_allclose(C_mat, expected_mat, rtol=1e-5)
    print("✓ VLIW decorator matrix multiply verification successful")


def test_vliw_decorator_conv2d():
    """Test the @vliw.kernel decorator with 2D convolution"""

    # Define function with @vliw.kernel decorator and custom indices
    @vliw.kernel(input_idx=[0, 1], output_idx=[2])
    def conv2d(input_img: float32[4, 4], kernel: float32[3, 3]) -> float32[2, 2]:
        output: float32[2, 2] = 0.0

        # Perform convolution with valid padding
        for i in allo.grid(2):  # Output height
            for j in allo.grid(2):  # Output width
                sum_val: float32 = 0.0
                for ki in allo.grid(3):  # Kernel height
                    for kj in allo.grid(3):  # Kernel width
                        sum_val += input_img[i + ki, j + kj] * kernel[ki, kj]
                output[i, j] = sum_val

        return output

    # Use directly in AIE dataflow
    @df.region()
    def conv2d_aie_design():
        @df.kernel(mapping=[1])
        def core(
            Input: float32[4, 4],
            Kernel: float32[3, 3],
            Output: float32[2, 2],
        ):
            conv2d(Input, Kernel, Output)

    conv2d_aie_module = df.build(conv2d_aie_design, target="aie")
    Input_img = np.random.rand(4, 4).astype(np.float32)
    Kernel_weights = np.random.rand(3, 3).astype(np.float32)
    Output_conv = np.zeros((2, 2)).astype(np.float32)
    conv2d_aie_module(Input_img, Kernel_weights, Output_conv)

    # verify the result with native numpy implementation
    def conv2d_reference(input_img, kernel):
        output = np.zeros((2, 2), dtype=np.float32)
        for i in range(2):
            for j in range(2):
                output[i, j] = np.sum(input_img[i : i + 3, j : j + 3] * kernel)
        return output

    expected_conv = conv2d_reference(Input_img, Kernel_weights)
    np.testing.assert_allclose(Output_conv, expected_conv, rtol=1e-5)
    print("✓ VLIW decorator 2D convolution verification successful")


if __name__ == "__main__":
    # Original approach tests
    # test_simple_add()
    # test_matrix_multiply()
    # test_conv2d()

    # New decorator approach test
    # test_simple_add_with_decorator()
    # test_vliw_decorator_matrix_multiply()
    test_vliw_decorator_conv2d()
    print("All tests passed! The new @vliw.kernel decorator simplifies VLIW usage.")
