import allo
from allo.ir.types import int32
import numpy as np
# M, N, K = 1024, 1024, 1024

def gemm(A: int32[64, 64], B: int32[64, 64]) -> int32[M, N]:
    C: int32[64, 64] = 0.0
    for i, j in allo.grid(64, 64):
        for k in allo.reduction(64):
            C[i, j] += A[i, k] * B[k, j]
    return C

s = allo.customize(gemm)
mod =s.build(target="vitis_hls", mode="hw", project="gemm_ncubed.prj")
print(s.module)