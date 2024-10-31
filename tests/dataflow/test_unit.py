# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import int32, Stream
import allo.dataflow as df
import re

M, N, K = 16, 16, 16
Mt, Nt = 1, 1
P0, P1 = Mt + 2, Nt + 2


@df.kernel(mapping=[P0, P1])
def gemm(A: int32[M, K], B: int32[K, N], C: int32[M, N]):
    # A[Mt, K] * B[K, Nt] = C[Mt, Nt]
    i, j = df.get_pid()
    in_A: Stream[int32] = df.pipe(src=(i, j - 1), dst=(i, j))
    in_B: Stream[int32] = df.pipe(src=(i - 1, j), dst=(i, j))
    out_A: Stream[int32] = df.pipe(src=(i, j), dst=(i, j + 1))
    out_B: Stream[int32] = df.pipe(src=(i, j), dst=(i + 1, j))
    for m in range(M // Mt):
        for n in range(N // Nt):
            with allo.meta_if(i in {0, Mt + 1, Nt + 1} and j in {0, Mt + 1, Nt + 1}):
                pass
            with allo.meta_elif(j == 0):
                for k in range(K):
                    out_A.put(A[m * Mt + i - 1, k])
            with allo.meta_elif(i == 0):
                for k in range(K):
                    out_B.put(B[k, n * Nt + j - 1])
            # drain
            with allo.meta_elif(i == Mt + 1):
                for k in range(K):
                    b: int32 = in_B.get()
            with allo.meta_elif(j == Nt + 1):
                for k in range(K):
                    a: int32 = in_A.get()
            # main body
            with allo.meta_else():
                c: int32 = 0
                for k in range(K):
                    a: int32 = in_A.get()
                    b: int32 = in_B.get()
                    c += a * b
                    out_A.put(a)
                    out_B.put(b)
                C[m * Mt + i - 1, n * Nt + j - 1] = c


def check_function_arguments(code, kernel_name, arg_count):
    pattern = rf"{kernel_name}\((.*?)\);"
    matches = re.findall(pattern, code)
    for match in matches:
        args = match.split(",")
        assert (
            len(args) == arg_count
        ), f"Expected {arg_count} arguments for {kernel_name}, but found {len(args)}."


def test_unit():
    mod = df.build(gemm, target="vhls")
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
    test_unit()
