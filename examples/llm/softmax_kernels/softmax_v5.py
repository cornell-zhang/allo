import math, allo
import numpy as np
from allo.ir.types import float32, index
from allo.customize import Partition as partition


Ty = float32
MIN_FLOAT32:Ty = -3.402823466e+38  # minimum float32 value
L = 256
unroll_factor = 16 * (L//64)


def softmax_top(QK_in: Ty[L, L]) -> Ty[L, L]:   
    QK_out: Ty[L, L]
    for i_soft in allo.grid(L, name = "i_soft"):
        max_val = softmax_p1(QK_in, i_soft)
        exp_buf_1 = softmax_p2(QK_in, max_val, i_soft)
        inv = softmax_p3(exp_buf_1)
        softmax_p4(QK_out, exp_buf_1, inv, i_soft)
    return QK_out

def softmax_p1(QK_in: Ty[L, L], i_pos: index) -> Ty:
    local_max: Ty = MIN_FLOAT32
    for j1 in allo.grid(L, name = "j1"):
        if QK_in[i_pos, j1] > local_max:
            local_max = QK_in[i_pos, j1]
    return local_max

def softmax_p2(QK_in: Ty[L, L], max_val: Ty, i_pos: index) -> Ty[L]:
    local_max: Ty = max_val
    exp_buf_1: Ty[L]
    for j2 in allo.grid(L, name = "j2"):
        e:Ty = allo.exp(QK_in[i_pos, j2] - local_max)
        exp_buf_1[j2] = e
    return exp_buf_1

def softmax_p3(exp_buf: Ty[L]) -> Ty:
    row_total: Ty = 0.0
    for j3 in allo.grid(L, name = "j3"):
        row_total += exp_buf[j3]
    inv:Ty = 1.0 / row_total
    return inv

def softmax_p4(QK_out: Ty[L, L], exp_buf: Ty[L], inv: Ty, i_pos: index):
    for j4 in allo.grid(L, name = "result"):
        QK_out[i_pos, j4] = exp_buf[j4] * inv

def true_softmax(QK_in: Ty[L, L]) -> Ty[L, L]:
    exp_buf:Ty[L, L] = 0.0      # TEMP for exponentials
    QK_out: Ty[L, L] = 0.0
    max_vals: Ty[L] = MIN_FLOAT32
    rows_total: Ty[L] = 0.0
    invs: Ty[L] = 0.0
    # --- Kernel B: scan QK_in and update max_vals in a perfect 2D nest ------
    for ii in range(L):
        for jj in range(L):
            v:Ty = QK_in[ii, jj]
            m:Ty = max_vals[ii]                   # affine.load
            # this select lowers to arithmetic, not a branch
            max_vals[ii] = v if v > m else m   # affine.store
    # --- Kernel C: compute exponentials and row sums (another perfect nest) -
    for ii in range(L):
        rows_total[ii] = 0.0
        for jj in range(L):
            e:Ty = allo.exp(QK_in[ii, jj] - max_vals[ii])
            exp_buf[ii, jj] = e
            rows_total[ii] += e
    # --- Kernel D: normalise (one more perfect nest) ------------------------
    for ii in range(L):
        inv:Ty = 1.0 / rows_total[ii] #this does not catch a division by zero
        invs[ii] = inv
    for ii in range(L):
        for jj in range(L):
            QK_out[ii, jj] = exp_buf[ii, jj] * invs[ii]
    return QK_out

    
def test_function_equivalence():
    QK_in = np.random.rand(L, L).astype(np.float32)
    QK_out_base = np.zeros((L, L), dtype=np.float32)
    QK_out_opt = np.zeros((L, L), dtype=np.float32)
    #base_sch = allo.customize(true_softmax)

    s1 = allo.customize(softmax_p1)
    s2 = allo.customize(softmax_p2)
    s3 = allo.customize(softmax_p3)
    s4 = allo.customize(softmax_p4)
    # s1.pipeline("j1", initiation_interval=1)
    s2.pipeline("j2", initiation_interval=1)
    # s3.pipeline("j3", initiation_interval=1)
    # s4.pipeline("j4", initiation_interval=1)
    top_sch = allo.customize(softmax_top)

   
    top_sch.pipeline("i_soft", initiation_interval=2)
   

    top_sch.partition(top_sch.QK_in, partition_type=partition.Cyclic, dim=2, factor = 8)
     #if L is 64 partition by 16,  if l is 128 partition by 32, if L is 256 partition by 64
    top_sch.partition(top_sch.QK_out, partition_type=partition.Cyclic, dim=2, factor = unroll_factor)

    # compose the lower level schedules into top_sch
    top_sch.compose([s1, s2, s3, s4])
    #top_sch.pipeline("i_soft", initiation_interval=1)

    opt_mod = top_sch.build(target="vitis_hls", mode="csyn", project="softmax_v5.prj")()
    #base_mod = base_sch.build(target="vitis_hls", mode="sw_emu", project="sw_softmax_true.prj")(QK_in, QK_out_base)
    # np.testing.assert_allclose(QK_out_opt, QK_out_base,  rtol=1e-5, atol=1e-5)
    # print("passed functional simulation 1")

if __name__ == "__main__":
    test_function_equivalence()