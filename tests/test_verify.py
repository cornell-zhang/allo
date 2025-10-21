# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
import allo
from allo.ir.types import int32, float32


def test_gemm():
    def gemm(A: float32[32, 32], B: float32[32, 32]) -> float32[32, 32]:
        C: float32[32, 32] = 0
        for i, j, k in allo.grid(32, 32, 32, name="C"):
            C[i, j] += A[i, k] * B[k, j]
        return C

    s_orig = allo.customize(gemm)
    s = allo.customize(gemm)
    s.reorder("gemm:i", "gemm:j")

    verifier = allo.verify(s, s_orig)
    assert verifier


def test_nested_functions_2():
    M, K, N = 32, 32, 32

    def gemm(A: int32[M, K], B: int32[K, N], C: int32[M, N]) -> None:
        for i, j in allo.grid(M, N):
            for k in allo.reduction(K):
                C[i, j] += A[i, k] * B[k, j]

    def top(A: int32[M, K], B: int32[K, N]) -> int32[M, N]:
        C: int32[M, N] = 0
        gemm(A, B, C)
        return C

    s1 = allo.customize(gemm)
    s1.reorder("k", "j")
    s1.partition(s1.C, dim=2)
    s1.buffer_at(s1.C, axis="i")
    s1.pipeline("j")
    # Top-level
    s = allo.customize(top)
    s.compose(s1)

    allo.verify(s, s1)


def test_range_for():
    def kernel(A: int32[20]):
        for i in range(10):
            A[i] = i
        for i in range(10, 20):
            A[i] = i
        for i in range(0, 20, 2):
            A[i] = i * 2

    s = allo.customize(kernel)
    verifier = allo.verify(s, s)

    assert verifier


# test that ap_int types are correctly handled
def test_get_bit():
    def kernel(A: int32[10]) -> int32[10]:
        B: int32[10] = 0
        for i in range(10):
            B[i] = (A[i] + 1)[0]
        return B

    s = allo.customize(kernel)
    verifier = allo.verify(s, s)

    assert verifier


def test_fusion():
    def separate_gemm(A: float32[32, 32], B: float32[32, 32]) -> float32[32, 32]:
        C: float32[32, 32] = 0
        for i, j, k in allo.grid(32, 32, 32):
            C[i, j] += A[i, k] * B[k, j]
        return C

    def fused_gemm(A: float32[32, 32], B: float32[32, 32]) -> float32[32, 32]:
        C: float32[32, 32] = 0
        for i in range(32):
            for j in range(32):
                for k in range(32):
                    C[i, j] += A[i, k] * B[k, j]
        return C

    s1 = allo.customize(separate_gemm)
    s2 = allo.customize(fused_gemm)

    verifier = allo.verify(s1, s2)
    assert verifier


def test_memory_layout():
    def row_major(A: float32[32, 32]) -> float32[32, 32]:
        B: float32[32, 32]
        for i, j in allo.grid(32, 32):
            B[i, j] = A[i, j]
        return B

    def column_major(A: float32[32, 32]) -> float32[32, 32]:
        B: float32[32, 32]
        for j, i in allo.grid(32, 32):
            B[i, j] = A[i, j]
        return B

    s1 = allo.customize(row_major)
    s2 = allo.customize(column_major)

    verifier = allo.verify(s1, s2)
    assert verifier


def test_strength_reduction():
    def mul_by_2(A: int32[10]) -> int32[10]:
        B: int32[10]
        for i in range(10):
            B[i] = A[i] * 2

        return B

    def shift_left(A: int32[10]) -> int32[10]:
        B: int32[10]
        for i in range(10):
            B[i] = A[i] << 1

        return B

    s1 = allo.customize(mul_by_2)
    s2 = allo.customize(shift_left)

    verifier = allo.verify(s1, s2)
    assert verifier


def test_compose_non_equivalence():
    def gemm(A: float32[32, 32], B: float32[32, 32]) -> float32[32, 32]:
        C: float32[32, 32] = 0
        for i, j, k in allo.grid(32, 32, 32):
            C[i, j] += A[i, k] * B[k, j]
        return C

    def incorrect(C: float32[32, 32], bias: float32[32]) -> float32[32, 32]:
        D: float32[32, 32] = 0
        for i, j in allo.grid(32, 32):
            D[i, j] = C[i, j] - bias[j]  # incorrect function
        return D

    def top(
        A: float32[32, 32], B: float32[32, 32], bias: float32[32]
    ) -> float32[32, 32]:
        C = gemm(A, B)
        D = incorrect(C, bias)
        return D

    s_gemm = allo.customize(gemm)
    s_add_bias_wrong = allo.customize(incorrect)

    # compose into top
    s_top = allo.customize(top)
    s_top.compose(s_gemm)
    s_top.compose(s_add_bias_wrong)

    # negative test: should return false
    verifier = allo.verify(s_top, s_gemm)
    assert (
        not verifier
    ), "Verifier incorrectly claims equivalence when functions are different"

    verifier = allo.verify(s_top, s_add_bias_wrong)
    assert not verifier, "Verifier failed to detect incorrect transformation"


if __name__ == "__main__":
    pytest.main([__file__])
