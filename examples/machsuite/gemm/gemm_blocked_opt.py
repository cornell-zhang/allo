import allo
from allo.ir.types import int32
import numpy as np


def bbgemm(A: int32[64, 64], B: int32[64, 64]) -> int32[64, 64]:
    C: int32[64, 64] = 0  
    sum_value: int32[1] = 0  

    for i in range(0, 64, 8):
        for j in range(0, 64, 8):
            for k in range(0, 64, 8):
                for ii in range(i, (i + 8) if (i + 8) < 64 else 64):
                    for jj in range(j, (j + 8) if (j + 8) < 64 else 64):
                        sum_value[0] = 0  
                        for kk in range(k, (k + 8) if (k + 8) < 64 else 64):
                            sum_value[0] += A[ii, kk] * B[kk, jj]
                        C[ii, jj] += sum_value[0]
    return C

s = allo.customize(bbgemm)
mod =s.build(target="vivado_hls", mode="csynth", project="gemm_blocked.prj")
# print(s.module)