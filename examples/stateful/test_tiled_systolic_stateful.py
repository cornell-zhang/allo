# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import re
import allo
from allo.ir.types import int32, Stream, stateful
import allo.dataflow as df
import allo.backend.hls as hls
import numpy as np


# M, N, K = 512, 512, 512
# Mt, Nt = 16, 16
M, N, K = 16, 16, 16
Mt, Nt, Kt = 4, 4, 4
# M, N, K = 4, 4, 4
# Mt, Nt = 1, 1
P0, P1 = Mt + 2, Nt + 2


@df.region()
def top():
    fifo_A: Stream[int32, 4][P0, P1]
    fifo_B: Stream[int32, 4][P0, P1]

    @df.kernel(mapping=[P0, P1])
    def gemm(A: int32[M, K], B: int32[K, N], C: int32[M, N]):
        # A[Mt, K] * B[K, Nt] = C[Mt, Nt]
        i, j = df.get_pid()
        for m in range(M // Mt):
            for n in range(N // Nt):
                for k in range(K // Kt):
                    # peripheral kernels
                    with allo.meta_if(i in {0, Mt + 1} and j in {0, Nt + 1}):
                        pass
                    with allo.meta_elif(j == 0):
                        # i > 0
                        for kt in range(Kt):
                            fifo_A[i, j + 1].put(A[m * Mt + i - 1, k * Kt + kt])
                    with allo.meta_elif(i == 0):
                        # j > 0
                        for kt in range(Kt):
                            fifo_B[i + 1, j].put(B[k * Kt + kt, n * Nt + j - 1])
                    # drain
                    with allo.meta_elif(i == Mt + 1):
                        for kt in range(Kt):
                            b: int32 = fifo_B[i, j].get()
                    with allo.meta_elif(j == Nt + 1):
                        for kt in range(Kt):
                            a: int32 = fifo_A[i, j].get()
                    # main body
                    with allo.meta_else():
                        c: stateful(int32) = 0
                        if k == 0:
                            c = 0
                        for kt in range(Kt):
                            a: int32 = fifo_A[i, j].get()
                            b: int32 = fifo_B[i, j].get()
                            c += a * b
                            fifo_A[i, j + 1].put(a)
                            fifo_B[i + 1, j].put(b)
                        if k == (K // Kt) - 1:
                            C[m * Mt + i - 1, n * Nt + j - 1] = c


def check_function_arguments(code, kernel_name, arg_count):
    pattern = rf"{kernel_name}\((.*?)\);"
    matches = re.findall(pattern, code)
    for match in matches:
        args = match.split(",")
        assert (
            len(args) == arg_count
        ), f"Expected {arg_count} arguments for {kernel_name}, but found {len(args)}."


def test_tiled_systolic():
    A = np.random.randint(0, 10, (M, K)).astype(np.int32)
    B = np.random.randint(0, 10, (K, N)).astype(np.int32)
    C = np.zeros((M, N), dtype=np.int32)

    sim_mod = df.build(top, target="simulator")
    sim_mod(A, B, C)
    np.testing.assert_allclose(C, np.dot(A, B), atol=1e-5)
    print("Dataflow Simulator Passed!")

    mod = df.build(top)
    if hls.is_available("vitis_hls"):
        C = np.zeros((M, N), dtype=np.int32)
        mod(A, B, C)
        np.testing.assert_allclose(C, np.dot(A, B), atol=1e-5)
        print("Passed!")

    # test generated module
    global Mt, Nt, P0, P1
    Mt, Nt = 1, 1
    P0, P1 = Mt + 2, Nt + 2
    mod = df.build(top, target="vhls")
    code = mod.hls_code
    unused_kernels = {"gemm_0_0", "gemm_0_2", "gemm_2_0", "gemm_2_2"}
    for kernel in unused_kernels:
        assert kernel not in code, f"Expected {kernel} not in hls code"
    check_function_arguments(code, "gemm_0_1", 4)
    check_function_arguments(code, "gemm_1_0", 4)
    check_function_arguments(code, "gemm_1_1", 7)
    check_function_arguments(code, "gemm_1_2", 4)
    check_function_arguments(code, "gemm_2_1", 4)
    print("Passed!")


if __name__ == "__main__":
    # we need to set OMP_NUM_THREADS to a large number here for simulator
    os.environ["OMP_NUM_THREADS"] = "64"
    test_tiled_systolic()
    del os.environ["OMP_NUM_THREADS"]
