# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import allo
from allo.ir.types import int32, float32
from allo.passes import monitor_memory_usage


def test_add_same_float():
    def kernel(A: float32[32, 32]) -> float32[32, 32]:
        B = A + A
        C = B + B
        D = C + C
        E = D + D
        return E

    s = allo.customize(kernel, enable_tensor=True)
    print(s.module)
    mod = s.build()
    monitor_memory_table = monitor_memory_usage(mod.intermediate_module)
    print(monitor_memory_table)
    assert (
        "| %alloc    | [32, 32] | f32     |       32768 |           2 | 2              |"
        in monitor_memory_table
    )


def test_add_const_float():
    def kernel(A: float32[32, 32]) -> float32[32, 32]:
        B = A + 1
        C = B + 1
        D = C + 1
        E = D + 1
        return E

    s = allo.customize(kernel, enable_tensor=True)
    print(s.module)
    mod = s.build()
    monitor_memory_table = monitor_memory_usage(mod.intermediate_module)
    print(monitor_memory_table)
    assert (
        "| %alloc_1  | [32, 32] | f32     |       32768 |           2 | 2              |"
        in monitor_memory_table
    )


def test_calculate():
    def kernel(A: float32[32], B: float32[32]) -> float32[32]:
        c: float32[32] = 3
        return allo.exp(A * B + c)

    s = allo.customize(kernel, enable_tensor=True)
    print(s.module)
    mod = s.build()
    monitor_memory_table = monitor_memory_usage(mod.intermediate_module)
    print(monitor_memory_table)
    assert (
        "| %alloc_1 | [32]    | f32     |        1024 |           1 | 2              |"
        in monitor_memory_table
    )


def test_gemm_grid_for():
    def gemm(A: int32[32, 32], B: int32[32, 32]) -> int32[32, 32]:
        C: int32[32, 32] = 0
        for i, j, k in allo.grid(32, 32, 32):
            C[i, j] += A[i, k] * B[k, j]
        return C

    s = allo.customize(gemm)
    s.split("i", 8)
    s.split("j", 8)
    s.reorder("i.outer", "j.outer", "i.inner", "j.inner")
    mod = s.build()
    print(mod.intermediate_module)
    monitor_memory_table = monitor_memory_usage(mod.intermediate_module)
    print(monitor_memory_table)
    assert (
        "| %alloc   | [32, 32] | i32     |       32768 |           2 | 1              |"
        in monitor_memory_table
    )


if __name__ == "__main__":
    pytest.main([__file__])
