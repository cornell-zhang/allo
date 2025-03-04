# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import allo
from allo.ir.types import int32, float32, MemRefType
from allo._mlir.dialects import func as func_d
from allo._mlir.dialects import memref as memref_d
from allo._mlir.dialects import affine as affine_d
from allo.passes import dataflow_canonicalization_pass as dcp, _mlir_lower_pipeline
from allo._mlir.ir import WalkResult
from allo.customize import Schedule

from allo.ir.transform import find_func_in_module

def check_single_producer_single_consumer(module):
    spsc = True

    def checker(op):
        nonlocal spsc
        if isinstance(op.opview, (func_d.CallOp, memref_d.AllocOp)):
            for produced_val in op.results:
                if produced_val is None or not isinstance(
                    produced_val.type, MemRefType
                ):
                    return WalkResult(0)
                if len(produced_val.type.shape) == 0:
                    return WalkResult(0)
                consumers = sum(
                    1
                    for use in produced_val.uses
                    if isinstance(
                        use.owner,
                        (func_d.CallOp, memref_d.LoadOp, affine_d.AffineLoadOp),
                    )
                    # ignore if inside if-statement
                    and not isinstance(use.owner.parent.opview, affine_d.AffineForOp)
                )
                if consumers > 1:
                    print(op)
                    spsc = False
                    return WalkResult(0)
        return WalkResult(0)

    with module.context:
        for func in module.body.operations:
            if not isinstance(func, func_d.FuncOp):
                continue
            func.walk(checker)
    return spsc


def canonicalize(schedule: Schedule) -> Schedule:
    fn_name = schedule.top_func.name.value
    mod = _mlir_lower_pipeline(schedule.module, lower_linalg=True)
    f = find_func_in_module(mod, fn_name)
    return Schedule(
        dcp(mod),
        f,
        schedule.func_args,
        schedule.ip,
        schedule.ext_libs,
        schedule.inst_list,
    )


def test_single_producer_single_consumer():
    def producer() -> int32[10]:
        A: int32[10]
        for i in range(10):
            A[i] = i
        return A

    def consumer(A: int32[10]) -> int32[10]:
        B: int32[10]
        for i in range(10):
            B[i] = A[i] + 1
        return B

    def top() -> int32[10]:
        A = producer()
        return consumer(A)

    p = allo.customize(producer)
    c = allo.customize(consumer)

    s = allo.customize(top)
    s.compose([p, c])
    s = canonicalize(s)
    print(s.module)
    mod = s.build()
    res = mod()
    np.testing.assert_array_equal(res, np.arange(1, 11))
    assert check_single_producer_single_consumer(s.module)


def test_single_producer_multiple_consumers():
    def producer() -> int32[10]:
        A: int32[10]
        for i in range(10):
            A[i] = i + 1
        return A

    def consumer1(A: int32[10]) -> int32:
        sum: int32 = 0
        for i in range(10):
            sum += A[i]
        return sum

    def consumer2(A: int32[10]) -> int32:
        prod: int32 = 1
        for i in range(10):
            prod *= A[i]
        return prod

    def top() -> int32:
        A = producer()
        return consumer1(A) + consumer2(A)

    p = allo.customize(producer)
    c1 = allo.customize(consumer1)
    c2 = allo.customize(consumer2)

    s = allo.customize(top)
    print(s.module)
    s.compose([p, c1, c2])
    s = canonicalize(s)
    print(s.module)

    mod = s.build()
    res = mod()
    np.testing.assert_array_equal(
        res, np.sum(np.arange(1, 11)) + np.prod(np.arange(1, 11))
    )
    assert check_single_producer_single_consumer(s.module)


def test_single_kernel():
    def producer() -> (int32[10], int32[10]):
        A: int32[10]
        for i in range(10):
            A[i] = i

        B: int32[10]
        C: int32[10]
        for i in range(10):
            B[i] = A[i] + 1
            C[i] = A[i] + 1
        return B, C

    s = allo.customize(producer)
    s = canonicalize(s)
    print(s.module)
    mod = s.build()
    res1, res2 = mod()
    np.testing.assert_array_equal(res1, np.arange(1, 11))
    np.testing.assert_array_equal(res2, np.arange(1, 11))
    assert check_single_producer_single_consumer(s.module)


def test_nd_array():
    def producer() -> int32[10, 10]:
        A: int32[10, 10] = 0
        return A

    def consumer(A: int32[10, 10]) -> int32:
        sum: int32 = 0
        for i in range(10):
            for j in range(10):
                sum += A[i, j]
        return sum

    def top() -> int32:
        A = producer()
        return consumer(A) + consumer(A)

    p = allo.customize(producer)
    c = allo.customize(consumer)

    s = allo.customize(top)
    s.compose([p, c])
    print(s.module)

    s = canonicalize(s)
    print(s.module)
    mod = s.build()
    res = mod()
    np.testing.assert_array_equal(res, 0)
    assert check_single_producer_single_consumer(s.module)


def test_matmul_addition_condition1():
    """Checks that the matmul reduction loop is transformed correctly as described in the Stream-HLS paper (https://arxiv.org/pdf/2501.09118)."""

    def matmul_addition() -> float32[8, 8]:
        A: float32[8, 8] = 1.0
        B: float32[8, 8] = 2.0
        C: float32[8, 8] = 0.0
        D: float32[8, 8] = 3.0

        for i in range(8):
            for j in range(8):
                for k in range(8):
                    C[i, j] = C[i, j] + A[i, k] * B[k, j]

        E: float32[8, 8]
        for i in range(8):
            for j in range(8):
                E[i, j] = C[i, j] + D[i, j]

        return E

    s = allo.customize(matmul_addition)
    print(s.module)

    s = canonicalize(s)
    print(s.module)

    mod = s.build()
    res = mod()

    expected = (
        np.full((8, 8), 1.0, dtype=np.float32) @ np.full((8, 8), 2.0, dtype=np.float32)
        + 3.0
    )

    np.testing.assert_allclose(res, expected, rtol=1e-5)
    assert check_single_producer_single_consumer(s.module)


def test_matmul_addition_nested_condition1():
    def matrix_multiply(A: float32[8, 8], B: float32[8, 8]) -> float32[8, 8]:
        C: float32[8, 8] = 0
        for i in range(8):
            for j in range(8):
                for k in range(8):
                    C[i, j] = C[i, j] + A[i, k] * B[k, j]
        return C

    def matrix_add(C: float32[8, 8], D: float32[8, 8]) -> float32[8, 8]:
        E: float32[8, 8]
        for i in range(8):
            for j in range(8):
                E[i, j] = C[i, j] + D[i, j]
        return E

    def top() -> float32[8, 8]:
        A: float32[8, 8] = 1.0
        B: float32[8, 8] = 2.0
        D: float32[8, 8] = 3.0

        C = matrix_multiply(A, B)

        E1 = matrix_add(C, D)
        E2 = matrix_add(C, C)  # Use C twice to force split

        return E1

    mm = allo.customize(matrix_multiply)
    ma = allo.customize(matrix_add)
    s = allo.customize(top)

    s.compose([mm, ma])

    print("Before preprocessing:")
    print(s.module)

    s = canonicalize(s)

    print("After preprocessing:")
    print(s.module)

    mod = s.build()
    res = mod()

    expected = np.full((8, 8), 19.0, dtype=np.float32)
    np.testing.assert_allclose(res, expected, rtol=1e-5)

    assert check_single_producer_single_consumer(s.module)


if __name__ == "__main__":
    pytest.main([__file__])
