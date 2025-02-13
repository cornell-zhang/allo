import numpy as np
import pytest
import allo
from allo.passes import check_perfect_affine_kernel
from allo.ir.types import int32, float32


def test_detect_perfect_affine_kernel():
    def perfect_kernel() -> int32[4, 4]:
        C: int32[4, 4]
        for i in range(4):
            for j in range(4):
                C[i, j] = i + j
        return C

    s_perfect = allo.customize(perfect_kernel)
    s_perfect.build()

    is_perfect = check_perfect_affine_kernel(s_perfect.module)
    assert is_perfect, "Expected the perfect_kernel to be detected as perfect."


    def not_perfect_kernel(A: int32[4, 4]) -> int32[4, 4]:
        B: int32[4, 4]
        B[0, 0] = A[0, 0] + 42
        for i in range(4):
            for j in range(4):
                B[i, j] = A[i, j] * 2
        return B

    s_imperfect = allo.customize(not_perfect_kernel)
    s_imperfect.build()
    print("Imperfect kernel module:\n", s_imperfect.module)

    is_perfect = check_perfect_affine_kernel(s_imperfect.module)
    assert not is_perfect, "Expected the not_perfect_kernel to fail perfect-affine detection."

