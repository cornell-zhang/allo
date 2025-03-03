# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import allo
from allo.ir.types import int32, float32, MemRefType
from allo._mlir.dialects import func as func_d
from allo._mlir.dialects import memref as memref_d
from allo._mlir.dialects import affine as affine_d
from allo.passes import dataflow_canonicalization_pass as dcp
from allo._mlir.ir import WalkResult
from allo.customize import Schedule


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
                )
                if consumers > 1:
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
    return Schedule(
        dcp(schedule.module),
        schedule.top_func,
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


if __name__ == "__main__":
    pytest.main([__file__])
