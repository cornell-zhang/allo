# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.primitives.unify import unify
from allo.ir.types import int8
import pytest
import numpy as np


def test_simple_loop():
    L = 8

    def f1(A: int8[L]):
        for i in range(L):
            A[i] += 1

    def f2(A: int8[L]):
        for i in range(L):
            A[i] -= 1

    unified = unify(f1, f2, 1)
    llvm_mod = allo.LLVMModule(unified, "f1_f2_unified")
    allo_A = np.zeros((L), dtype=np.int8)
    np_A = allo_A.copy()
    np_A_add = allo_A + 1

    llvm_mod(allo_A, np.array([0], dtype=np.int8))
    np.testing.assert_allclose(allo_A, np_A_add, atol=1e-3)

    llvm_mod(allo_A, np.array([1], dtype=np.int8))
    np.testing.assert_allclose(allo_A, np_A, atol=1e-3)


def test_multi_time_loop():
    L = 8

    def f1(A: int8[L]):
        for i in range(L):
            A[i] += 1

    def f2(A: int8[L]):
        for i in range(L):
            A[i] -= 1

    unified = unify(f1, f2, 4)
    llvm_mod = allo.LLVMModule(unified, "f1_f2_unified")
    allo_A = np.zeros((L), dtype=np.int8)
    np_A_add = allo_A + 2

    llvm_mod(allo_A, np.array([0, 0, 0, 1], dtype=np.int8))
    np.testing.assert_allclose(allo_A, np_A_add, atol=1e-3)


def test_simple_grid():
    L, D = 8, 8

    def f1(A: int8[L, D]):
        for i, j in allo.grid(L, D):
            A[i, j] += 1

    def f2(A: int8[L, D]):
        for i, j in allo.grid(L, D):
            A[i, j] -= 1

    unified = unify(f1, f2, 1)
    llvm_mod = allo.LLVMModule(unified, "f1_f2_unified")
    allo_A = np.zeros((L, D), dtype=np.int8)
    np_A = allo_A.copy()
    np_A_add = allo_A + 1

    llvm_mod(allo_A, np.array([0], dtype=np.int8))
    np.testing.assert_allclose(allo_A, np_A_add, atol=1e-3)

    llvm_mod(allo_A, np.array([1], dtype=np.int8))
    np.testing.assert_allclose(allo_A, np_A, atol=1e-3)


def test_select():
    L = 4

    def f1(A: int8[L], B: int8[L], C: int8[L]):
        for i in range(L):
            A[i] = B[i]

    def f2(A: int8[L], B: int8[L], C: int8[L]):
        for i in range(L):
            A[i] = C[i]

    unified = unify(f1, f2, 1)
    llvm_mod = allo.LLVMModule(unified, "f1_f2_unified")
    allo_A = np.zeros(L, dtype=np.int8)
    allo_B = np.ones(L, dtype=np.int8)
    allo_C = np.full(L, 2, dtype=np.int8)

    llvm_mod(allo_A, allo_B, allo_C, np.array([0], dtype=np.int8))
    np.testing.assert_allclose(allo_A, allo_B, atol=1e-3)

    llvm_mod(allo_A, allo_B, allo_C, np.array([1], dtype=np.int8))
    np.testing.assert_allclose(allo_A, allo_C, atol=1e-3)


def test_nested_loop():
    L, D = 4, 4

    def f1(A: int8[L], B: int8[L]):
        for i in range(L):
            A[i] = B[i]
            for j in range(D):
                A[i] += 1

    def f2(A: int8[L], B: int8[L]):
        for i in range(L):
            A[i] = B[i]
            for j in range(D):
                A[i] -= 1

    unified = unify(f1, f2, 1)
    llvm_mod = allo.LLVMModule(unified, "f1_f2_unified")
    allo_A = np.zeros((L), dtype=np.int8)
    allo_B = np.array([5, 6, 7, 8], dtype=np.int8)

    llvm_mod(allo_A, allo_B, np.array([0], dtype=np.int8))
    np.testing.assert_allclose(
        allo_A, np.array([9, 10, 11, 12], dtype=np.int8), atol=1e-3
    )

    llvm_mod(allo_A, allo_B, np.array([1], dtype=np.int8))
    np.testing.assert_allclose(allo_A, np.array([1, 2, 3, 4], dtype=np.int8), atol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__])
