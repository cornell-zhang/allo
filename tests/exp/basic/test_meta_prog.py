# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from allo.exp import build, process
from allo.exp.passes.utils import unroll_meta_for
import allo
from allo.ir.types import int32
from allo.template import meta_for as allo_for
from allo.spmw import kernel


def test_meta_for():
    # [NOTE]: semantic test only, currently cannot unroll the loop
    @kernel
    def kernel1(A: int32[20]) -> int32[20]:
        with allo.meta_for(10) as i:
            A[i] = i
        return A

    s = process(kernel1)
    np_A = np.zeros((20,), dtype=np.int32)
    gold = np.zeros((20,), dtype=np.int32)
    for i in range(10):
        gold[i] = i
    assert np.allclose(s(np_A), gold)

    @kernel
    def kernel2(A: int32[20]) -> int32[20]:
        with allo.meta_for(10) as i:
            A[i + 1] = i
        return A

    s = process(kernel2)
    np_A = np.zeros((20,), dtype=np.int32)
    gold = np.zeros((20,), dtype=np.int32)
    for i in range(10):
        gold[i + 1] = i
    assert np.allclose(s(np_A), gold)

    @kernel
    def kernel3(A: int32[20]) -> int32[20]:
        with allo.meta_for(10) as i:
            A[i * 2] = i
        return A

    s = process(kernel3)
    np_A = np.zeros((20,), dtype=np.int32)
    gold = np.zeros((20,), dtype=np.int32)
    for i in range(10):
        gold[i * 2] = i
    assert np.allclose(s(np_A), gold)

    @kernel
    def kernel4(A: int32[20]):
        with allo.template.meta_for(10) as i:
            A[i] = i
        with allo_for(10, 20) as i:
            A[i] = i
        with allo.meta_for(0, 20, 2) as i:
            A[i] = i * 2

    s = process(kernel4)
    np_A = np.zeros((20,), dtype=np.int32)
    gold = np.zeros((20,), dtype=np.int32)
    for i in range(10):
        gold[i] = i
    for i in range(10, 20):
        gold[i] = i
    for i in range(0, 20, 2):
        gold[i] = i * 2
    s(np_A)
    assert np.allclose(np_A, gold)

    @kernel
    def kernel5(A: int32[20]) -> int32[20]:
        with allo.meta_for(10) as i:
            with allo.meta_for(2) as j:
                A[i * 2 + j] = i
        return A

    s = process(kernel5)
    np_A = np.zeros((20,), dtype=np.int32)
    gold = np.zeros((20,), dtype=np.int32)
    for i in range(10):
        for j in range(2):
            gold[i * 2 + j] = i
    assert np.allclose(s(np_A), gold)

    def get_constant(value: int):
        return value

    @kernel
    def kernel6(A: int32[20]) -> int32[20]:
        with allo.meta_for(get_constant(10)) as i:
            with allo.meta_for(get_constant(2)) as j:
                A[i * 2 + j] = i
        return A

    s = process(kernel6)
    np_A = np.zeros((20,), dtype=np.int32)
    gold = np.zeros((20,), dtype=np.int32)
    for i in range(10):
        for j in range(2):
            gold[i * 2 + j] = i
    assert np.allclose(s(np_A), gold)

    print("test_meta_for passed")


def test_unroll_meta_for():

    @kernel
    def kernel1(A: int32[20]) -> int32[20]:
        with allo.meta_for(10) as i:
            A[i] = i
        return A

    module, top_name = build(kernel1)
    assert "affine.for" in str(module) and '"unroll"' in str(module)
    unroll_meta_for(module)
    print(module)
    assert "affine.for" not in str(module)

    @kernel
    def kernel2(A: int32[20]) -> int32[20]:
        with allo.meta_for(4) as i:
            with allo.meta_for(5) as j:
                A[i * 5 + j] = i
        return A

    module, top_name = build(kernel2)
    print(module)
    assert "affine.for" in str(module) and '"unroll"' in str(module)
    unroll_meta_for(module)
    assert "affine.for" not in str(module)

    @kernel
    def kernel3(A: int32[20]) -> int32[20]:
        inc: int32 = 0
        with allo.meta_for(4) as i:
            with allo.meta_for(5) as j:
                A[i * 5 + j] = i
            inc += i
        return A

    module, top_name = build(kernel3)
    print(module)
    assert "affine.for" in str(module) and '"unroll"' in str(module)
    unroll_meta_for(module)
    assert "affine.for" not in str(module)


if __name__ == "__main__":
    # test_meta_for()
    test_unroll_meta_for()
