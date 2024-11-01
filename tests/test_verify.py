import pytest
import numpy as np
import allo
from allo.ir.types import int32, float32

def test_gemm():
    def gemm(A: float32[32, 32], B: float32[32, 32]) -> float32[32, 32]:
        C: float32[32, 32] = 0
        # Use grid_for with name annotation
        for i, j, k in allo.grid(32, 32, 32, name="C"):
            C[i, j] += A[i, k] * B[k, j]
        return C

    s_orig = allo.customize(gemm)
    s = allo.customize(gemm)
    s.reorder("gemm:i", "gemm:j")
    f = s.build()

    verifier = allo.verify(s, s_orig)
    assert verifier


# def test_nested_functions_2():
#     M, K, N = 32, 32, 32

#     def gemm(A: int32[M, K], B: int32[K, N], C: int32[M, N]) -> None:
#         for i, j in allo.grid(M, N):
#             for k in allo.reduction(K):
#                 C[i, j] += A[i, k] * B[k, j]

#     def top(A: int32[M, K], B: int32[K, N]) -> int32[M, N]:
#         C: int32[M, N] = 0
#         gemm(A, B, C)
#         return C

#     s1 = allo.customize(gemm)
#     s1.reorder("k", "j")
#     s1.partition(s1.C, dim=2)
#     s1.buffer_at(s1.C, axis="i")
#     s1.pipeline("j")
#     # Top-level
#     s = allo.customize(top)
#     s.compose(s1)

#     allo.verify(s, s1)


if __name__ == "__main__":
    pytest.main([__file__])

