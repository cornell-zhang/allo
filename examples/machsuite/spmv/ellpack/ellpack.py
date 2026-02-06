import allo
import numpy as np
from allo.ir.types import int32, float64

# Generate a random size for a matrix
r = np.random.randint(1, 10)
c = np.random.randint(1, 10)

N = 494 # Number of rows
L = 10 # Number of non zero entries in row

def ellpack(NZ: float64[N*L], cols: int32[N*L], vec: float64[N]) -> float64[N]:
    out: float64[N] = 0

    for i, j in allo.grid(N, L):
        if  cols[j + i*L] != -1: # For rows with fewer than L non zero entries
            out[i] += NZ[j + i*L] * vec[cols[j + i*L]]
    
    return out

s = allo.customize(ellpack)
mod = s.build()
# print(s.module)



## Testing
# Generating Random Sparse Matrix
upperbound = []
for i in range(r):
    upperbound.append([np.random.randint(1,np.random.randint(2,10))])

rMatrix = np.random.randint(0, high=upperbound, size=(len(upperbound), c)).astype(np.int32)

# Generating Random Vector
vector = np.random.randint(0, 100, (c)).astype(np.int32)


# Applying Ellpack Format
def ellpack_format(matrix):
    values = []
    columns = []


    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] != 0:
                columns.append(j)
            else:
                columns.append(-1)
            values.append(matrix[i][j])

    values = np.array(values).astype(np.float64)
    columns = np.array(columns).astype(np.int32)

    return (values, columns)

values, columns = ellpack_format(rMatrix)

# expected = np.matmul(rMatrix, vector)
# observed = mod(values, columns, vector)

# np.testing.assert_allclose(observed, expected, rtol=1e-5, atol=1e-5)





