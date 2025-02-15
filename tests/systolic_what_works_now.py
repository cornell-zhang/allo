
import numpy as np
import allo
from allo.ir.types import int8, int16, index

# What's working right now

def PE_kernel(
    A_in: int8[4],
    B_in: int8[4],
    A_out: int8[4],
    B_out: int8[4],
    C: int16[4, 4],
    i: index,
    j: index,
):
    v: int16 = 0
    for k in range(4):
        a: int8 = A_in[k]
        b: int8 = B_in[k]
        v += a * b
        A_out[k] = a
        B_out[k] = b
    C[i, j] = v


def systolic_tile(
    A: int8[4, 4],
    B: int8[4, 4],
    C: int16[4, 4],
):
    A_fifo: int8[4, 4 + 1, 4]
    B_fifo: int8[4, 4 + 1, 4]
    A_drain: int8[4]
    B_drain: int8[4]

    for k in range(4, name="data_load"):
        for i in range(4):
            A_fifo[i, 0, k] = A[i, k]
        for j in range(4):
            B_fifo[j, 0, k] = B[k, j]
    
    for i, j in allo.grid(4, 4, name="PE"):
        PE_kernel(A_fifo[i, j], B_fifo[j, i], A_fifo[i, j + 1], B_fifo[j, i + 1], C, i, j)

    for k in range(4, name="data_drain"):
        for i in range(4):
            A_drain[i] = A_fifo[i, 4, k]
        for j in range(4):
            B_drain[j] = B_fifo[j, 4, k]


s = allo.customize(systolic_tile)
print(s.module)
s.partition(s.C, dim=0)
s.partition(s.A, dim=1)
s.partition(s.B, dim=2)
pe = s.unfold("PE", [0, 1])
# s.to(s.A_fifo, pe, axis=1, depth=4 + 1)
# s.to(s.B_fifo, pe, axis=0, depth=4 + 1)

# s.build(target="vitis_hls", mode="hw_emu", project="systolic.prj")
mod = s.build()
A = np.random.randint(0, 10, size=(4, 4), dtype=np.int8)
B = np.random.randint(0, 10, size=(4, 4), dtype=np.int8)
C = np.zeros((4, 4), dtype=np.int16)
mod(A, B, C)
print(C)
np_C = A.astype(np.int32) @ B.astype(np.int32)
print(np_C)
