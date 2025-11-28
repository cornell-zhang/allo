import math, allo
import numpy as np
from allo._mlir import ir as mlir_ir
StringAttr = mlir_ir.StringAttr
from allo.ir.types import float32, int32, index
from allo.customize import Partition as partition
from allo.ir.utils import MockBuffer

TILE_SIZE = 16
L = 64
unroll_factor = L//TILE_SIZE
Ty = float32
MIN_FLOAT32:Ty = -3.402823466e+38  # minimum float32 value

def softmax_top(QK_in: Ty[L, L]) -> Ty[L, L]:     # TEMP for exponentials
    QK_out: Ty[L, L] = 0.0
    i_arr_1: int32[L*L]

    for i in allo.grid(L*L, name = "i"):
        x: index = i
        i_arr_1[i] = x

    for i_outer in allo.grid(L//TILE_SIZE, name = "i_outer"):
        for i_inner in allo.grid(TILE_SIZE, name = "i_inner"):
            max_val = softmax_p1(QK_in, i_arr_1[i_outer*TILE_SIZE + i_inner])
            exp_buf_1 = softmax_p2(QK_in, max_val, i_arr_1[i_outer*TILE_SIZE + i_inner])
            inv = softmax_p3(exp_buf_1)
            softmax_p4(QK_out, exp_buf_1, inv, i_arr_1[i_outer*TILE_SIZE + i_inner])
    return QK_out

def softmax_p1(QK_in: Ty[L, L], i_pos: int32) -> Ty:
    local_max: Ty = MIN_FLOAT32
    for j1 in allo.grid(L, name = "j1"):
        if QK_in[i_pos, j1] > local_max:
            local_max = QK_in[i_pos, j1]
    return local_max

def softmax_p2(QK_in: Ty[L, L], max_val: Ty, i_pos: int32) -> Ty[L]:
    local_max: Ty = max_val
    #exp_buf: Ty[L] = 0.0
    exp_buf_1: Ty[L] = 0.0
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

def softmax_p4(QK_out: Ty[L, L], exp_buf: Ty[L], inv: Ty, i_pos: int32):
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
    base_sch = allo.customize(true_softmax)

    s1 = allo.customize(softmax_p1)
    s2 = allo.customize(softmax_p2)
    s3 = allo.customize(softmax_p3)
    s4 = allo.customize(softmax_p4)
    top_sch = allo.customize(softmax_top)

    top_sch.partition(top_sch.QK_in, partition_type=partition.Cyclic, dim=1, factor = unroll_factor)
    top_sch.partition(top_sch.i_arr_1, partition_type=partition.Cyclic, dim=1, factor = 16)
    # top_sch.partition(top_sch.i_arr_2, partition_type=partition.Cyclic, dim=1, factor = 16)
    # top_sch.partition(top_sch.i_arr_3, partition_type=partition.Cyclic, dim=1, factor = 16)
    # top_sch.partition(top_sch.i_arr_4, partition_type=partition.Cyclic, dim=1, factor = 16)
    top_sch.partition(top_sch.QK_out, partition_type=partition.Cyclic, dim=2, factor = 32)

    # compose the lower level schedules into top_sch
    top_sch.compose([s1, s2, s3, s4])

    top_sch.unroll("i", 16)
    top_sch.pipeline("i")

    base_mod = base_sch.build(target="vitis_hls", mode="sw_emu", project="sw_softmax_true.prj")(QK_in, QK_out_base)
    opt_mod = top_sch.build(target="vitis_hls", mode="sw_emu", project="sw_softmax_top_opt.prj")(QK_in, QK_out_opt)
    np.testing.assert_allclose(QK_out_opt, QK_out_base,  rtol=1e-5, atol=1e-5)
    print("passed functional simulation 1")

def test_true_softmax():
    s = allo.customize(true_softmax)
    mod = s.build(target="vitis_hls", mode="csyn", project="true_softmax.prj")()

def test_softmax_top():
    # create the schedules
    s1 = allo.customize(softmax_p1)
    s2 = allo.customize(softmax_p2)
    s3 = allo.customize(softmax_p3)
    s4 = allo.customize(softmax_p4)
    top_sch = allo.customize(softmax_top)

    # # create fifo streams from the inner loop to softmax_p4
    # top_sch.to(top_sch.exp_buf, "softmax_p4")
    # top_sch.to(top_sch.invs, "softmax_p4")

    # partitions
    top_sch.partition(top_sch.QK_in, partition_type=partition.Cyclic, dim=0, factor = unroll_factor)
    #top_sch.partition(top_sch.i_arr_1, partition_type=partition.Cyclic, dim=1, factor = 16)
    #top_sch.partition(top_sch.invs, partition_type=partition.Cyclic, dim=1, factor = unroll_factor*2)
    #top_sch.partition(top_sch.i_arr, partition_type=partition.Cyclic, dim=1, factor = 16)
    #top_sch.partition(top_sch.exp_buf, partition_type=partition.Cyclic, dim=1, factor = 16)
    top_sch.partition(top_sch.QK_out, partition_type=partition.Cyclic, dim=0, factor = 8)

    # schedule the lower level functions
    # schedule_softmax_p1(s1)
    # schedule_softmax_p2(s2)
    # schedule_softmax_p3(s3)
    # schedule_softmax_p4(s4)
    
    # compose the lower level schedules into top_sch
    top_sch.compose([s1, s2, s3, s4])

    top_sch.unroll("i", 16)
    top_sch.pipeline("i")

    # unfold the inner loop and apply dataflow to the outer loop
    #top_sch.dataflow(top_sch.get_loops("softmax_top")["i_outer"]["i_inner"])
    #top_sch.unroll("i_outer", factor = 2)
    #top_sch.unfold("i_outer", [0]) #unfold the outer loop
    #top_sch.dataflow("softmax_top")
    
    # build the top level schedule
    mod = top_sch.build(target="vitis_hls", mode="csyn", project="softmax_top_8part.prj")()


def schedule_softmax_p1(s):
    ### partitions are just here for bookkeeping ###
    # s.partition(s.QK_in, partition_type=partition.Cyclic, dim=1, factor = 4)
    s.pipeline("j1")
    

def schedule_softmax_p2(s):
    ### partitions ###
    #s.partition(s.exp_buf, partition_type=partition.Cyclic, dim=1, factor = 4)
    #s.unroll("j2", factor = 4)
    s.pipeline("j2")


def schedule_softmax_p3(s):
    ### partitions are just here for bookkeeping ###
    #s.partition(s.exp_buf, partition_type=partition.Cyclic, dim=1, factor = 4)
    #s.unroll("j3", factor = 4)
    s.pipeline("j3")


def schedule_softmax_p4(s):
    ### partitions are just here for bookkeeping ###
    # s.partition(s.QK_out, partition_type=partition.Cyclic, dim=1, factor = 16)
    #s.partition(s.exp_buf, partition_type=partition.Cyclic, dim=1, factor = 16)
    s.unroll("j4", factor = 16)
    s.pipeline("j4")

def schedule_base():
    s1 = allo.customize(softmax_p1)
    s2 = allo.customize(softmax_p2)
    s3 = allo.customize(softmax_p3)
    s4 = allo.customize(softmax_p4)
    top_sch = allo.customize(softmax_top)
    top_sch.compose([s1, s2, s3, s4])
    return top_sch


def test_base():
    top_sch = schedule_base()
    mod = top_sch.build(target="vitis_hls", mode="csyn", project="softmax_top_base.prj")()



if __name__ == "__main__":
    #test_base()
    test_softmax_top()
    #test_true_softmax()
    #test_function_equivalence()