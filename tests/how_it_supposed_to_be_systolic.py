import numpy as np
import allo
from allo.ir.types import int8, int16, index


def gemm(
    A: int8[4, 4],
    B: int8[4, 4],
    C: int16[4, 4]
):
    for i, j in allo.grid(4, 4, name="PE"):
        for k in range(4):
            C[i, j] += A[i, k] * B[k, j]


s = allo.customize(gemm)
s.prepare_systolic("PE")
# print(s.module)
pe = s.unfold("PE", [0, 1])
s.partition(s.C, dim=0)
s.partition(s.A, dim=1)
s.partition(s.B, dim=2)
s.to(s.A_fifo, pe, axis=1, depth=4 + 1)
s.to(s.B_fifo, pe, axis=0, depth=4 + 1)
mod = s.build(target="vitis_hls", mode="csyn", project="new_systolic.prj")
# mod = s.build(target="vitis_hls", mode="csim", project="new_systolic_csim.prj")
# np_A = np.random.randint(0, 10, size=(4, 4)).astype(np.int16)
# np_B = np.random.randint(0, 10, size=(4, 4)).astype(np.int16)
# np_C = np.matmul(np_A, np_B)
# np_C_allo = np.zeros((4, 4), dtype=np.int16)
# mod(np_A, np_B, np_C_allo)
# np.testing.assert_allclose(np_C, np_C_allo, rtol=1e-5)
# print("Passed!")


# # mod = s.build()
# A = np.random.randint(0, 10, size=(4, 4), dtype=np.int8)
# B = np.random.randint(0, 10, size=(4, 4), dtype=np.int8)
# C = np.zeros((4, 4), dtype=np.int16)
# mod(A, B, C)
# print(C)
# np_C = A.astype(np.int32) @ B.astype(np.int32)
# print(np_C)