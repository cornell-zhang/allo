# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import allo
from allo.ir.types import float32, int32, Stream
import allo.dataflow as df
from allo.backend import hls

IR, IC = 6, 6  # Input rows and columns
FR, FC = 3, 3  # Filter rows and columns
OR, OC = IR - FR + 1, IC - FC + 1  # Output rows and columns

P0, P1 = (
    OR * OC + 2,
    FR,
)  # We need a PE per element in the output matrix, and two layers of PE to add up results


# Base convolution kernel (truth that we compare against)
def conv2D_lb(A: float32[IR, IC], B: float32[FR, FC]) -> float32[OR, OC]:
    C: float32[OR, OC] = 0
    for y, x in allo.grid(OR, OC):  # these are the output dimensions
        v: float32 = 0
        for r, c in allo.reduction(FR, FC):  # this is the filter dimensions
            v += A[y + r, x + c] * B[FR - r - 1, FC - c - 1]
        C[y, x] = v
    return C


# Convolution kernel with systolic array (basic)
@df.region()
def top():
    fifo_A: Stream[float32, FR * FC][P0, P1]
    fifo_B: Stream[float32, FR * FC][P0, P1]

    @df.kernel(mapping=[P0, P1])
    def conv_kernel(A: float32[IR, IC], B: float32[FR, FC], C: float32[OR, OC]):
        pi, pj = df.get_pid()

        # We do not use these PEs
        with allo.meta_if(pi in {0, P0 - 1} and pj in {0, P1 - 1}):
            pass

        # This meta_if loads the input matrix into the PEs
        with allo.meta_elif(pj == 0):
            output_row: int32 = 0
            output_col: int32 = 0

            for r in range(OR):
                if (pi > r * OC) and (pi <= (r + 1) * OC):
                    output_row = r
                    output_col = pi - (r * OC) - 1
            for row, col in allo.grid(FR, FC):
                fifo_A[pi, pj + 1].put(A[row + output_row, col + output_col])

        # This meta_elif loads the filter matrix into the PEs
        with allo.meta_elif(pi == 0):
            for row, col in allo.grid(FR, FC):
                fifo_B[pi + 1, pj].put(B[FR - row - 1, FC - col - 1])
                # We do this because we want the last entry of the filter matrix to go first into the PE

        # These next two meta_elif load the partial sums into the drain PEs
        with allo.meta_elif(pi == P0 - 1 and pj > 0):
            for col in range(FR * FC):
                drain_B: float32 = fifo_B[pi, pj].get()

        with allo.meta_elif(pj == P1 - 1 and pi > 0):
            for row in range(FR * FC):
                drain_A: float32 = fifo_A[pi, pj].get()

        # This meta_else does the main multiplication of the convolution kernel
        with allo.meta_else():
            partial_sum: float32 = 0
            for k in range(FR * FC):
                a: float32 = fifo_A[pi, pj].get()
                b: float32 = fifo_B[pi, pj].get()
                partial_sum += a * b
                fifo_A[pi, pj + 1].put(partial_sum)
                fifo_B[pi + 1, pj].put(b)

            out_row: int32 = 0
            out_col: int32 = 0
            for r in range(OR):
                if (pi > r * OC) and (pi <= (r + 1) * OC):
                    out_row = r
                    out_col = pi - r * OC - 1
            C[out_row, out_col] += partial_sum


# Testing the systolic convolution kernel
def test_convolution():
    # Build ground truth kernel
    s = allo.customize(conv2D_lb)
    test_mod = s.build()

    # Build systolic convolution kernel
    sim_mod = df.build(top, target="simulator")
    print("Test and systolic models built")

    # Random test cases
    for i in range(100):
        A = np.random.rand(IR, IC).astype(np.float32)
        B = np.random.rand(FR, FC).astype(np.float32)
        C_sys = np.zeros((OR, OC), dtype=np.float32)
        test_C = np.zeros((OR, OC), dtype=np.float32)

        test_C = test_mod(A, B)
        sim_mod(A, B, C_sys)

        np.testing.assert_allclose(C_sys, test_C, atol=1e-3)

    print("Simulation passed!")

    mod = df.build(top, target="vitis_hls", mode="hw_emu")
    if hls.is_available("vitis_hls"):
        C_sys = np.zeros((OR, OC), dtype=np.float32)
        mod(A, B, C_sys)
        np.testing.assert_allclose(C_sys, test_C, atol=1e-5)
        print("Passed!")


if __name__ == "__main__":
    test_convolution()
