# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import allo
from allo.ir.types import float32

V = 10000


# This test is using a vector add kernel
def vector_add(A: float32[V], B: float32[V]) -> float32[V]:
    C: float32[V] = 0.0
    for i in allo.grid(V):
        C[i] = A[i] + B[i]
    return C


s = allo.customize(vector_add)
s.pipeline("i")

project_name = "test_in_test74"

code = s.build(target="ihls", mode="source_file_only", project=project_name)

kernel_file_path = os.path.join(project_name, "kernel.cpp")

try:
    with open(kernel_file_path, "r", encoding="utf-8") as kernel_file:
        content = kernel_file.read()
        search_string0 = "Automatically generated file for Intel High-level Synthesis (HLS)"  # Test if the file is created by checking header
        search_string1 = "queue q(selector, dpc_common::exception_handler);"  # Test if a queue is created
        search_string2 = "h.single_task<Top>([=]() [[intel::kernel_args_restrict]]"  # Test if the kernel is created

        if (
            search_string0 in content
            and search_string1 in content
            and search_string2 in content
        ):
            print(
                f"\033[92mTest Case: 3 test cases passed in {kernel_file_path}\033[0m"
            )
        else:
            raise RuntimeError(
                f"\033[91mTest Case: failed in {kernel_file_path}\033[0m"
            )

except FileNotFoundError:
    print(f"\033[91mTest Case: The file {kernel_file_path} does not exist.\033[0m")
