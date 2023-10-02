# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import allo
from allo.ir.types import int32, float32
from allo.passes import monitor_memory_usage


def test_add_same_int():
    def kernel(A: int32[32, 32]) -> int32[32, 32]:
        B = A + A
        C = B + B
        D = C + C
        E = D + D
        return E

    s = allo.customize(kernel, enable_tensor=True)
    print(s.module)
    mod = s.build()
    monitor_memory_table = monitor_memory_usage(s.module, mod.intermediate_module, True)
    print(monitor_memory_table)


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
    monitor_memory_table = monitor_memory_usage(s.module, mod.intermediate_module, True)
    print(monitor_memory_table)


def test_add_const_int():
    def kernel(A: int32[32, 32]) -> int32[32, 32]:
        B = A + 1
        C = B + 1
        D = C + 1
        E = D + 1
        return E

    s = allo.customize(kernel, enable_tensor=True)
    print(s.module)
    mod = s.build()
    monitor_memory_table = monitor_memory_usage(s.module, mod.intermediate_module, True)
    print(monitor_memory_table)


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
    monitor_memory_table = monitor_memory_usage(s.module, mod.intermediate_module, True)
    print(monitor_memory_table)


if __name__ == "__main__":
    pytest.main([__file__])
