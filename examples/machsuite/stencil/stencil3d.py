import allo
import numpy as np
from allo.ir.types import int32

height_size = 32
col_size = 32
row_size = 16

def stencil3d(C: int32[2], orig: int32[row_size, col_size, height_size]) -> int32[row_size, col_size, height_size]:
    sol: int32[row_size, col_size, height_size] = 0
    sum0: int32 = 0
    sum1: int32 = 0
    mul0: int32 = 0
    mul1: int32 = 0

    # Handle boundary conditions by filling with original values
    for j, k in allo.grid(col_size, row_size):
        sol[k, j, 0] = orig[k, j, 0]
        sol[k, j, height_size - 1] = orig[k, j, height_size - 1]

    for i, k in allo.grid(height_size - 1, row_size):
        sol[k, 0, (i+1)] = orig[k, 0, (i+1)]
        sol[k, col_size - 1, (i+1)] = orig[k, col_size - 1, (i+1)]

    for j, i in allo.grid(col_size-2, height_size-2):
        sol[0, (j+1), (i+1)] = orig[0, (j+1), (i+1)]
        sol[row_size - 1, (j+1), (i+1)] = orig[row_size - 1, (j+1), (i+1)]

    # Stencil computation
    for i, j, k in allo.grid( height_size - 2, col_size - 2, row_size - 2 ):
        sum0 = orig[(k+1), (j+1), (i+1)]
        sum1 = (orig[(k+1), (j+1), (i+2)] +
                orig[(k+1), (j+1), i] +
                orig[(k+1), (j+2), (i+1)] +
                orig[(k+1), j, (i+1)] +
                orig[(k+2), (j+1), (i+1)] +
                orig[k, (j+1), (i+1)])
        mul0 = sum0 * C[0]
        mul1 = sum1 * C[1]
        sol[(k+1), (j+1), (i+1)] = mul0 + mul1

    return sol

s = allo.customize(stencil3d)
mod = s.build(target="llvm")

np.random.seed(42)
np_C = np.random.randint(1, 5, size=2).astype(np.int32)
np_orig = np.random.randint(0, 100, (row_size, col_size, height_size)).astype(np.int32)

np_sol = mod(np_C, np_orig)

# Python reference
ref_sol = np.zeros((row_size, col_size, height_size), dtype=np.int32)

# Boundary: top/bottom height planes
for j in range(col_size):
    for k in range(row_size):
        ref_sol[k, j, 0] = np_orig[k, j, 0]
        ref_sol[k, j, height_size - 1] = np_orig[k, j, height_size - 1]

# Boundary: front/back col planes
for i in range(height_size - 1):
    for k in range(row_size):
        ref_sol[k, 0, i+1] = np_orig[k, 0, i+1]
        ref_sol[k, col_size - 1, i+1] = np_orig[k, col_size - 1, i+1]

# Boundary: left/right row planes
for j in range(col_size - 2):
    for i in range(height_size - 2):
        ref_sol[0, j+1, i+1] = np_orig[0, j+1, i+1]
        ref_sol[row_size - 1, j+1, i+1] = np_orig[row_size - 1, j+1, i+1]

# Interior stencil
for i in range(height_size - 2):
    for j in range(col_size - 2):
        for k in range(row_size - 2):
            s0 = np_orig[k+1, j+1, i+1]
            s1 = (np_orig[k+1, j+1, i+2] +
                  np_orig[k+1, j+1, i] +
                  np_orig[k+1, j+2, i+1] +
                  np_orig[k+1, j, i+1] +
                  np_orig[k+2, j+1, i+1] +
                  np_orig[k, j+1, i+1])
            ref_sol[k+1, j+1, i+1] = s0 * np_C[0] + s1 * np_C[1]

np.testing.assert_allclose(np_sol, ref_sol, rtol=1e-5, atol=1e-5)
print("PASS!")
