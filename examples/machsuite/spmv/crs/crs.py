import numpy as np
import allo
from allo.ir.types import int32, float64

N = 494  # Number of rows
NNZ = 1666 # Number of nonzero values

def crs(val: float64[NNZ], cols: int32[NNZ], row: int32[N+1], vec: float64[N]) -> float64[N]:
    out: float64[N] = 0.0

    for i in range(N):
        tmp_begin: int32 = row[i]
        tmp_end: int32 = row[i+1]

        for j in range(tmp_begin, tmp_end):
            out[i] += val[j] * vec[cols[j]]

    return out
