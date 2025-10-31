import math, allo
import numpy as np
from allo._mlir import ir as mlir_ir
from allo.ir.types import float32, int32, index

L, D = 64, 1024
Ty = float32

# def top(Q: Ty[L, D])->Ty[D]:
#     Q_new: Ty[D]  # Declare outside the loop
#     for i in allo.grid(1, name = "row_loop"):
#         Q_new_tmp = middle(Q)
#         for j in allo.grid(D, name = "col_loop"):
#             Q_new[j] = Q_new_tmp[j]
#     return Q_new

def middle(Q: Ty[L, D])->Ty[D]:
    Q_row: Ty[D]
    Q_row_new: Ty[D]  # Declare outside the loop
    for i in allo.grid(L, name = "row_loop"):
        for j in allo.grid(D, name = "col_loop"):
            Q_row[j] = Q[i, j]
        Q_tmp = bottom(Q_row)
        for j in allo.grid(D, name = "col_loop"):
            Q_row_new[j] = Q_tmp[j]
    return Q_row_new

def bottom(Q: Ty[D]) -> Ty[D]:
    Q_row: Ty[D]
    for i in allo.grid(D, name = "row_loop"):
        Q_row[i] = Q[i]
    return Q_row

def test_tiling():
    Q = np.random.rand(L, D).astype(np.float32)

    s1 = allo.customize(bottom)
    s2 = allo.customize(middle)
    #s3 = allo.customize(top)
    s2.buffer_at(s2.Q_row, "i")

    s2.compose([s1])
    #s3.compose([s2])
    mod = s2.build(target = "vitis_hls", mode = "csyn", project = "test_tiling_copy.prj")()

if __name__ == "__main__":
    test_tiling()
