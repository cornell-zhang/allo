import numpy as np
import allo
from allo.ir.types import int32

# Generate a random size for matrix
r = np.random.randint(0,100)
c = np.random.randint(0,50)

N = 494  # Number of rows
NNZ = 1666 # Number of nonzero values (Placeholder)

def crs(val: float64[NNZ], cols: int32[NNZ], row: int32[N+1], vec: float64[N]) -> float64[N]:
    out: float64[N] = 0.0

    for i in range(N):
        tmp_begin: int32 = row[i]
        tmp_end: int32 = row[i+1]

        for j in range(tmp_begin, tmp_end):
            out[i] += val[j] * vec[cols[j]]

    return out

s = allo.customize(crs)
mod = s.build()







## Testing 

# Arbirtrarily generated matrix
rMatrix = np.random.randint(0, 9, [r, c]).astype(np.float64)

# Arbitrarily generated vector
vector = np.random.randint(0, 100, (c)).astype(np.float64)

# CRS Format
def crs_format(matrix):
    values = []
    columns = []
    row = [0]

    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] != 0:
                values.append(matrix[i][j])
                columns.append(j)
        row.append(len(values))

    values = np.array(values).astype(np.float64)
    columns = np.array(columns).astype(np.int32)
    row = np.array(row).astype(np.int32)


    return (values, columns, row)


(values, columns, row) = crs_format(rMatrix)

# Calculations
# observed = mod(values, columns, row, vector)
# expected = np.matmul(rMatrix, vector)

# np.testing.assert_allclose(observed, expected, rtol=1e-5, atol=1e-5)






