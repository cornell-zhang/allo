import math, allo
import numpy as np
from allo._mlir import ir as mlir_ir
StringAttr = mlir_ir.StringAttr
from allo.ir.types import float32, int32, index
from allo.autoscheduler.passes import dataflow_optimization_pass
from allo.autoscheduler.config import AutoschedulerConfig
from allo.autoscheduler.dfg import DFG
from gurobipy import GurobiError
from allo.customize import Partition as partition
from allo.ir.utils import MockBuffer
from allo.ir.transform import find_buffer

# --------------------------------------------------------------------------------
# Problem size: 1 head of 32 × 32 with 64-wide model dimension
# (easy to synthesize quickly; raise L/D/H for bigger experiments)
# --------------------------------------------------------------------------------
H, L, D = 12, 512, 768
P = H // 12     # parallel heads
h_d:int32 = D // H
Ty = float32
MIN_FLOAT32:Ty = -3.402823466e+38  # minimum float32 value
TILE_SIZE = 16
R1 = 8
R2 = 16 * (L//64)

softmax_p3_row_totals = 8 ## this is the depth of the pipeline in softmax_p3

import math
import numpy as np
float32_scalar = float32        

def gemm(A: Ty[L, P*h_d], B: Ty[L, P*h_d], start_pos: int32) -> Ty[L, L]:
    """Is the AB^T technically where if B is already transposed it would be A[L, start_pos:start_pos+D]B[start_pos:start_pos+D, L]
    this is like computing Q_hK_h^T with causal masking and normalization
    """
    C: Ty[L, L] = MIN_FLOAT32
    for i in range(L, name="gemm_transpose_outer_loop"):
        for j in range(L, name="gemm_transpose_inner_loop"):
            for k in allo.reduction(h_d):
                C[i, j] = A[i, start_pos + k] * B[j, start_pos + k]
    return C

def gemm_2(A: Ty[L, L], B: Ty[L, P*h_d], start_pos: int32) -> Ty[L, h_d]:
    C: Ty[L, h_d]
    for i in range(L):
        for j in range(h_d):
            D: Ty = 0.0
            for k in allo.reduction(L):
                D += A[i, k] * B[k, start_pos + j]
            C[i, j] = D
    return C

def softmax_top(QK_in: Ty[L, L]) -> Ty[L, L]:     # TEMP for exponentials
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
    row_totals: Ty[softmax_p3_row_totals] = 0.0
    total_sum: Ty = 0.0
    row_totals_index: int32 = 0
    for j3 in allo.reduction(L, name = "j3"):
        row_totals_index = j3 % softmax_p3_row_totals
        row_totals[row_totals_index] += exp_buf[j3]
    for k3 in allo.reduction(softmax_p3_row_totals):
        total_sum += row_totals[k3]
    inv:Ty = 1.0 / total_sum
    return inv

def softmax_p4(QK_out: Ty[L, L], exp_buf: Ty[L], inv: Ty, i_pos: index):
    for j4 in allo.grid(L, name = "result"):
        QK_out[i_pos, j4] = exp_buf[j4] * inv

# ------------------------------------------------------------------------------
def attention_parallel_subset(Q_sliced: Ty[L, P*h_d], K_sliced: Ty[L, P*h_d], V_sliced: Ty[L, P*h_d])->Ty[L, P*h_d]:
    Z_i: Ty[L, P*h_d]
    start_pos:int32 = 0
    # full_start_position: int32 = 0
    Q_h: Ty[L, h_d]
    K_h: Ty[L, h_d]
    V_h: Ty[L, h_d]
    for j in allo.grid(P, name="multi_head_inner_loop"):
        for ii, jj in allo.grid(L, h_d, name="loop_smaller_slicing"):
            Q_h[ii, jj] = Q_sliced[ii, j * h_d + jj]
            K_h[ii, jj] = K_sliced[ii, j * h_d + jj]
            V_h[ii, jj] = V_sliced[ii, j * h_d + jj]
        QK_t = gemm(Q_h, K_h)
        QK_t_s = softmax_top(QK_t)
        Z_i_j = gemm_2(QK_t_s, V_h)
        for (i2, j2) in allo.grid(L, h_d, name="store_intermediate_result"):
            Z_i[i2, start_pos + j2] = Z_i_j[i2, j2]
    return Z_i

def attention_parallel_full(Q: Ty[L, D], K: Ty[L, D], V: Ty[L, D]) -> Ty[L, D]:
    Z: Ty[L, D]
    Q_sliced: Ty[L, P*h_d]
    K_sliced: Ty[L, P*h_d]
    V_sliced: Ty[L, P*h_d]
    for i in allo.grid(H//P, name="multi_head_outer_loop"):
        for ii, jj in allo.grid(L, P*h_d, name="loop_slicing"):
            Q_sliced[ii, jj] = Q[ii, i*P*h_d + jj]
            K_sliced[ii, jj] = K[ii, i*P*h_d + jj]
            V_sliced[ii, jj] = V[ii, i*P*h_d + jj]
        Z_new = attention_parallel_subset(Q_sliced, K_sliced, V_sliced)
        for (i2, j2) in allo.grid(L, P*h_d, name="store_final_result"):
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
    my_solution = np.zeros(Q.shape)
    solution = sdp(Q, K, V, H, D)
    
    s1 = allo.customize(gemm)
    s2 = allo.customize(gemm_2)
    soft_top = allo.customize(softmax_top)
    s5 = allo.customize(attention_parallel_subset)

    soft_1 = allo.customize(softmax_p1)
    soft_2 = allo.customize(softmax_p2)
    soft_3 = allo.customize(softmax_p3)
    soft_4 = allo.customize(softmax_p4)

    soft_1.pipeline("j1", initiation_interval=1)
    soft_2.pipeline("j2", initiation_interval=1)
    soft_3.pipeline("j3", initiation_interval=1)
    soft_3.pipeline("k3", initiation_interval=1)
    soft_4.pipeline("j4", initiation_interval=1)

    soft_top.compose([soft_1, soft_2, soft_3, soft_4])
    #soft_top.pipeline("i_soft", initiation_interval=3)

    s1.reorder("k", "j")
    s1.buffer_at(s1.C, "i")
    s1.pipeline("j", initiation_interval=1)
    s1.unroll("j", factor=8)

    s2.pipeline("j", initiation_interval=1)
    #s2.pipeline("k3", initiation_interval=1)
    #s2.unroll("k", factor=16)

    s5.partition("attention_parallel_subset:QK_t", partition.Cyclic, dim=2, factor=R1)
    s5.partition("attention_parallel_subset:QK_t_s", partition.Cyclic, dim=2, factor=R2)

    s5.compose([s1, s2, soft_top])
    s5.unfold("multi_head_inner_loop", [0])

    s6 = allo.customize(attention_parallel_full)

    s6.partition(s6.Q, partition.Block, dim=2, factor=H)
    s6.partition(s6.K, partition.Block, dim=2, factor=H)
    s6.partition(s6.V, partition.Block, dim=2, factor=H)
    s6.partition(s6.Z, partition.Block, dim=2, factor=H)
    s6.partition(s6.Q_sliced, partition.Block, dim=0, factor=P)
    s6.partition(s6.K_sliced, partition.Block, dim=0, factor=P)
    s6.partition(s6.V_sliced, partition.Block, dim=0, factor=P)
    
    s6.compose([s5])
    mock_buffer_z_new = MockBuffer("attention_parallel_full", "Z_new")
    s6.partition(mock_buffer_z_new, partition.Block, dim=2, factor=P)
    s6.dataflow("attention_parallel_subset")
    s6.dataflow("attention_parallel_full")
    mode = "csyn"
    #s6.build(target="vitis_hls", mode="csyn", project=f"att_partial_par_{P}_heads_{H}.prj")()
    s6.build(target="vitis_hls", mode=mode, project=f"unroll_{P}.prj")()
    #s6.build(target="vitis_hls", mode="sw_emu", project=f"sw_no_iarr_unroll_{P}_pipeline_{R1}_{R2}.prj")(Q, K, V, my_solution)
    #print(f"unrolled {R1} and {R2} in softmax")
    print("softmax_p3 modified")
    # print(my_solution)
    # print("-"*100)
    # print(solution)
    # np.testing.assert_allclose(my_solution, solution, atol=1e-5)
    # print("everything passed!")


if __name__ == "__main__":
    test_attention()