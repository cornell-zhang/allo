# test_packed_systolic.py
import allo
from allo.ir.types import int8, int16, int32
import allo.dataflow as df
import allo.backend.hls as hls
import numpy as np

L, D = 2, 2
M, N, K = L, 1*D, D
PP = 2
P0, P1 = M // PP + 2, N + 2

if PP == 2:
    np_type = np.int16
    allo_type = int16
else:
    raise ValueError(f"Unsupported packing factor: {PP}")

@df.region()
def top():
    fifo_A = df.array(df.pipe(dtype=allo_type, shape=(), depth=4), shape=(P0, P1))
    fifo_B = df.array(df.pipe(dtype=allo_type, shape=(), depth=4), shape=(P0, P1))

    @df.kernel(mapping=[P0, P1])
    def gemm(
        X_packed: allo_type[L // PP, D],
        W_packed: allo_type[D, 1 * D // PP],
        Z_packed: allo_type[L // PP, 1 * D]
    ):
        i, j = df.get_pid()
        # Peripheral kernels
        with allo.meta_if(i in {0, M + 1} and j in {0, N + 1}):
            pass
        with allo.meta_elif(j == 0):
            # i > 0
            for k in range(K // PP):
                fifo_A[i, j + 1].put(X_packed[i - 1, k])
        with allo.meta_elif(i == 0):
            # j > 0
            for k in range(K // PP):
                fifo_B[i + 1, j].put(W_packed[k, (j - 1)])
        # drain
        with allo.meta_elif(i == M + 1 and j > 0):
            for k in range(K // PP):
                b: allo_type = fifo_B[i, j].get()
        with allo.meta_elif(j == N + 1 and i > 0):
            for k in range(K // PP):
                a: allo_type = fifo_A[i, j].get()
        # main body
        with allo.meta_else():
            c: allo_type = 0
            for k in range(K):
                a: allo_type = fifo_A[i, j].get()
                b: allo_type = fifo_B[i, j].get()
                for p in range(PP):
                    a_unpacked: int8 = a[p * 8 : (p + 1) * 8]
                    b_unpacked: int8 = b[p * 8 : (p + 1) * 8]
                    c[p * 8 : (p + 1) * 8] += a_unpacked * b_unpacked
                fifo_A[i, j + 1].put(a)
                fifo_B[i + 1, j].put(b)
                Z_packed[i - 1, j - 1][k * 8 : (k + 1) * 8] += c

def test_packed_systolic():
    X = np.random.randint(-4, 4, size=(L, D)).astype(np.int8)
    print("X:")
    print(X)
    W_A_cst = np.random.randint(-4, 4, size=(D, 1 * D)).astype(np.int8)
    print("W_A_cst:")
    print(W_A_cst)
    
    packed_X = np.ascontiguousarray(
        np.ascontiguousarray(X).view(np_type).transpose()
    )
    print("packed_X:")
    print(packed_X)
    W_A_packed = np.ascontiguousarray(
        np.ascontiguousarray(W_A_cst.transpose()).view(np_type).transpose()
    )
    print("W_A_packed")
    print(W_A_packed)
    Z_packed = np.zeros((L // PP, 1 * D), dtype=np_type)
    mod = df.build(top)
    if hls.is_available("vitis_hls"):
        mod(packed_X, W_A_packed, Z_packed)
        np_C = X @ W_A_cst
        print("np_C:")
        print(np_C)
        np_C_packed = np.ascontiguousarray(
            np.ascontiguousarray(np_C).view(np_type).transpose()
        )
        print("np_C_packed:")
        print(np_C_packed)
        np.testing.assert_allclose(Z_packed, np_C_packed, atol=1e-3)
        print("Passed!")

if __name__ == "__main__":
    test_packed_systolic()
