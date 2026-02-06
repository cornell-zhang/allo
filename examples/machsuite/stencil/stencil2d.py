import allo
import numpy as np
from allo.ir.types import int32

col_size = 64
row_size = 128
f_size = 9

def stencil2d(orig: int32[row_size, col_size], filter: int32[f_size] ) -> int32[row_size, col_size]:
    sol: int32[row_size, col_size] = 0
    for i, j in allo.grid(row_size-2, col_size-2):
        temp: int32= 0
        for m, n in allo.grid(3, 3):
            mul: int32= filter[m*3 + n] * orig[(i+m), (j+n)]
            temp += mul
        sol[i, j] = temp
    return sol

if __name__ == "__main__":
    s = allo.customize(stencil2d)
    mod = s.build(target="llvm")

    np.random.seed(42)
    np_orig = np.random.randint(0, 100, (row_size, col_size)).astype(np.int32)
    np_filter = np.random.randint(0, 10, (f_size,)).astype(np.int32)

    np_sol = mod(np_orig, np_filter)

    # Python reference
    ref_sol = np.zeros((row_size, col_size), dtype=np.int32)
    for r in range(row_size - 2):
        for c in range(col_size - 2):
            temp = 0
            for k1 in range(3):
                for k2 in range(3):
                    temp += np_filter[k1 * 3 + k2] * np_orig[r + k1, c + k2]
            ref_sol[r, c] = temp

    np.testing.assert_allclose(np_sol, ref_sol, rtol=1e-5, atol=1e-5)
    print("PASS!")
