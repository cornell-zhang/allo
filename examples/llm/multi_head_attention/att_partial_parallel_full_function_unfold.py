import math, allo
import numpy as np
from allo._mlir import ir as mlir_ir
StringAttr = mlir_ir.StringAttr
from allo.ir.types import float32, int32, Int, index
from allo.autoscheduler.passes import dataflow_optimization_pass
from allo.autoscheduler.config import AutoschedulerConfig
from allo.autoscheduler.dfg import DFG
from gurobipy import GurobiError
from allo.customize import Partition as partition
from allo.ir.utils import MockBuffer

# --------------------------------------------------------------------------------
# Problem size: 1 head of 32 × 32 with 64-wide model dimension
# (easy to synthesize quickly; raise L/D/H for bigger experiments)
# --------------------------------------------------------------------------------
H, L, D = 8, 64, 1024
P = H // 1 # parallel heads
h_d:int32 = D // H
Ty = float32
MIN_FLOAT32:Ty = -3.402823466e+38  # minimum float32 value

TILE_SIZE_SOFTMAX = 16

import math
import numpy as np
float32_scalar = float32        

def gemm(A: float32[L, D], B: float32[L, D], start_pos: int32) -> float32[L, L]:
    """Is the AB^T technically where if B is already transposed it would be A[L, start_pos:start_pos+D]B[start_pos:start_pos+D, L]
    this is like computing Q_hK_h^T
    """
    C: float32[L, L] = 0.0
    for i in range(L, name="gemm_transpose_outer_loop"):
        for j in range(L, name="gemm_transpose_inner_loop"):
            for k in allo.reduction(h_d):
                C[i, j] += A[i, start_pos + k] * B[j, start_pos + k]
    return C

def gemm_2(A: float32[L, L], B: float32[L, D]) -> float32[L, h_d]:
    C: float32[L, h_d] = 0.0
    for i in range(L):
        for j in range(h_d):
            for k in allo.reduction(L):
                C[i, j] += A[i, k] * B[k, j]
    return C
def softmax_top(QK_in: Ty[L, L]) -> Ty[L, L]:     # TEMP for exponentials
    QK_out: Ty[L, L] = 0.0
    i_arr_1: int32[L*L]

    for i in allo.grid(L*L, name = "i"):
        x: index = i
        i_arr_1[i] = x

    for i_outer in allo.grid(L//TILE_SIZE_SOFTMAX, name = "i_outer"):
        for i_inner in allo.grid(TILE_SIZE_SOFTMAX, name = "i_inner"):
            max_val = softmax_p1(QK_in, i_arr_1[i_outer*TILE_SIZE_SOFTMAX + i_inner])
            exp_buf_1 = softmax_p2(QK_in, max_val, i_arr_1[i_outer*TILE_SIZE_SOFTMAX + i_inner])
            inv = softmax_p3(exp_buf_1)
            softmax_p4(QK_out, exp_buf_1, inv, i_arr_1[i_outer*TILE_SIZE_SOFTMAX + i_inner])
    return QK_out

def softmax_p1(QK_in: Ty[L, L], i_pos: int32) -> Ty:
    local_max: Ty = MIN_FLOAT32
    for j1 in allo.grid(L, name = "j1"):
        if QK_in[i_pos, j1] > local_max:
            local_max = QK_in[i_pos, j1]
    return local_max

def softmax_p2(QK_in: Ty[L, L], max_val: Ty, i_pos: int32) -> Ty[L]:
    local_max: Ty = max_val
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

# ------------------------------------------------------------------------------
def attention_parallel_subset(Q: Ty[L, D], K: Ty[L, D], V: Ty[L, D])->Ty[L, P*h_d]:
    Z_i: Ty[L, P*h_d]
    start_pos:int32 = 0
    for j in allo.grid(P, name="multi_head_inner_loop"):
        start_pos = j * h_d
        QK_t = gemm(Q, K, start_pos)
        QK_t_s = softmax_top(QK_t)
        Z_i_j = gemm_2(QK_t_s, V)
        for (i2, j2) in allo.grid(L, h_d):
            Z_i[i2, start_pos + j2] = Z_i_j[i2, j2]
    return Z_i


def attention_parallel_full(Q: Ty[L, D], K: Ty[L, D], V: Ty[L, D]) -> Ty[L, D]:
    Z: Ty[L, D]
    for i in allo.grid(H//P, name="multi_head_outer_loop"):
        Z_new = attention_parallel_subset(Q, K, V)
        for (i2, j2) in allo.grid(L, P*h_d):
            Z[i2, i*P*h_d + j2] = Z_new[i2, j2]
    return Z

def np_softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def sdp(Q, K, V, H, D):
    context = np.zeros(Q.shape)
    h_d = D // H
    for i in range(H):
        # split Q, K, V
        Q_h = Q[:, i * h_d : (i + 1) * h_d]
        K_h = K[:, i * h_d : (i + 1) * h_d]
        V_h = V[:, i * h_d : (i + 1) * h_d]
        # compute attention
        attention = np.matmul(Q_h, K_h.T)
        Y = np_softmax(attention)
        context_i = np.matmul(Y, V_h)
        context[:, i * h_d : (i + 1) * h_d] = context_i
    return context


def test_attention():
    Q = np.random.rand(L, D).astype(np.float32)
    K = np.random.rand(L, D).astype(np.float32)
    V = np.random.rand(L, D).astype(np.float32)
    # solution = sdp(Q, K, V, H, D)
    
    # call .customize() on all kernels
    s1 = allo.customize(gemm)
    s2 = allo.customize(gemm_2)
    soft_top = allo.customize(softmax_top)
    s5 = allo.customize(attention_parallel_subset)
    soft_1 = allo.customize(softmax_p1)
    soft_2 = allo.customize(softmax_p2)
    soft_3 = allo.customize(softmax_p3)
    soft_4 = allo.customize(softmax_p4)
    soft_top = allo.customize(softmax_top)

    ### first we make the softmax kernel ###
    # partitions
    #soft_top.partition(soft_top.QK_in, partition_type=partition.Cyclic, dim=0, factor = 4)
    soft_top.partition(soft_top.i_arr_1, partition_type=partition.Cyclic, dim=1, factor = 16)
    #soft_top.partition(soft_top.QK_out, partition_type=partition.Cyclic, dim=0, factor = 16)

    # compose the lower level softmax schedules into soft_sch
    soft_top.compose([soft_1, soft_2, soft_3, soft_4])
    # 
    soft_top.unroll("i", 16)
    soft_top.pipeline("i")
    soft_top.unroll("i_outer", factor = 4)
    # s5.partition(s5.Z_i, partition.Block, dim=2, factor=P)

    s1.reorder("k", "j")
    s1.buffer_at(s1.C, "i")
    s1.pipeline("k")
    s1.unroll(axis="j", factor=H)

    s5.partition("attention_parallel_subset:QK_t", partition.Cyclic, dim=0, factor=4)
    s5.partition("attention_parallel_subset:QK_t_s", partition.Cyclic, dim=0, factor=16)

    #s5.partition(MockBuffer("attention_parallel_subset", s5.get_loops("attention_parallel_subset")["multi_head_inner_loop"]["j"]:"Qk_t"), partition.Cyclic, dim=0, factor=4)
    #s5.get_loops("attention_parallel_subset")["multi_head_inner_loop"]["j"], "Qk_t"), partition.Cyclic, dim=0, factor=4)
    #s5.partition(MockBuffer(s5.get_loops("attention_parallel_subset")["multi_head_inner_loop"]["j"], "Qk_t_s"), partition.Cyclic, dim=0, factor=16)
    #s5.partition(MockBuffer("attention_parallel_subset", "Qk_t_s"), partition.Block, dim=0, factor=16)

    s5.compose([s1, s2, soft_top])
    s5.unfold("multi_head_inner_loop", [0])

    s6 = allo.customize(attention_parallel_full) 
    s6.partition(s6.Q, partition.Block, dim=2, factor=H)
    s6.partition(s6.K, partition.Block, dim=2, factor=H)
    s6.partition(s6.V, partition.Block, dim=2, factor=H)
    s6.compose([s5])
    s5.partition(s5.Z_i, partition.Block, dim=2, factor=P)
    s6.dataflow("attention_parallel_subset")

    s6.build(target="vitis_hls", mode="csyn", project=f"att_partial_par_{P}_heads_{H}.prj")()
    print("everything passed!")


if __name__ == "__main__":
    test_attention()