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
# P = H // 4 # parallel heads
# h_d:int32 = D // H
Ty = float32
MIN_FLOAT32 = -3.402823466e+38  # minimum float32 value

TILE_SIZE_SOFTMAX = 16

import math
import numpy as np
float32_scalar = float32    

def flash_attention(Q: Ty[L, D], K: Ty[L, D], V: Ty[L, D])->Ty[L, D]:
    Z: Ty[L, D]
    Q_row: Ty[D]
    # K_row: Ty[D]
    # V_row: Ty[D]
    for i in allo.grid(L, name = "unroll_row_loop"):
        for j in allo.grid(D, name= "row_tile_loop"):
            Q_row[j] = Q[i, j]
        Z_row = flash_attention_tile(Q_row, K, V)
        for j in allo.grid(D, name = "j"):
            Z[i, j] = Z_row[j]
    return Z


# def flash_attention_row(Q_row: Ty[D], K: Ty[L, D], V: Ty[L, D])->Ty[D]:
#     # K_row: Ty[D]
#     # V_row: Ty[D]
#     # for y in allo.grid(L, name = "y"):
#     #     for k in allo.grid(D, name = "k"):
#     #         K_row[k] = K[i, k]  #kiT
#     #         V_row[k] = V[i, k]  #Vi


#         Z_row = flash_attention_tile(Q_row, K_row, V_row)

#     # for i in allo.grid(D, name = "flash_attention_tile_loop"): # iterate over the columns of one row
#     #     qkT = dot_product(Q_row, K_row)
#     #     Z_row = online_softmax(qkT, V_row,)

#     return Z_row


def flash_attention_tile(Q_row: Ty[D], K: Ty[L, D], V: Ty[L, D])->Ty[D]:
    local_max_curr: Ty = MIN_FLOAT32
    local_max_prev: Ty = MIN_FLOAT32
    exp_prev: Ty = 0.0
    exp_curr: Ty = 0.0
    Z_prev: Ty[D] = 0.0
    Z_curr: Ty[D]

    for i in allo.grid(L, name = "flash_attention_tile_loop"):
        # dot product
        qkT: Ty = 0.0
        for j1 in allo.grid(D, name = "dot_product"):
            qkT += Q_row[j1] * K[i, j1]

        # online softmax
        if qkT > local_max_prev:
            local_max_curr = qkT
        exp_curr = allo.exp(local_max_prev - local_max_curr)*exp_prev + allo.exp(qkT - local_max_curr)
        for j2 in allo.grid(D, name = "softmax_main_loop"):
            tmp1: Ty = Z_prev[j2]*(exp_prev/exp_curr)*allo.exp(local_max_prev - local_max_curr) 
            tmp2: Ty = (allo.exp(qkT - local_max_curr)/exp_curr)*V[i, j2]
            Z_curr[j2] = tmp1 + tmp2  

        local_max_prev = local_max_curr
        exp_prev = exp_curr

        for j3 in allo.grid(D, name = "cur_prev_update_loop"):
            Z_prev[j3] = Z_curr[j3]

    return Z_curr



def dot_product(Q_row: Ty[D], K_row: Ty[D])->Ty:
    #this kernel computes the dot product of qkT vectors
    total: Ty = 0.0
    for j in allo.grid(D, name = "dot_product"):
        total += Q_row[j] * K_row[j]
    return total

def online_softmax(qkT: Ty, V_row: Ty[D]) -> Ty[D]:

    if qkT > local_max_prev:
        local_max_curr = qkT
    exp_curr = allo.exp(local_max_prev - local_max_curr)*exp_prev + allo.exp(qkT - local_max_curr)
    for i in allo.grid(D, name = "softmax_main_loop"):
        tmp_1 = Z_prev[i]*(exp_prev/exp_curr)*allo.exp(local_max_prev - local_max_curr) 
        tmp_2 = (allo.exp(qkT - local_max_curr)/exp_curr)*V_row[i]
        Z_curr[i] = tmp_1 + tmp_2

    local_max_prev = local_max_curr
    exp_prev = exp_curr

    for i in allo.grid(D, name = "cur_prev_update_loop"):
        Z_prev[i] = Z_curr[i]

    return Z_curr

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

    
def test_flash_attention():
    Q = np.random.rand(L, D).astype(np.float32)
    K = np.random.rand(L, D).astype(np.float32)
    V = np.random.rand(L, D).astype(np.float32)
    my_solution = np.zeros(Q.shape)
    solution = sdp(Q, K, V, H, D)

    s1 = allo.customize(flash_attention_tile)
    s2 = allo.customize(flash_attention)
    s2.compose([s1])

    # mod = s4.build()
    # my_sol = mod(Q, K, V)

    #mod = s3.build(target = "vitis_hls", mode = "csyn", project = "flash_attention.prj")()
    #mod = s2.build(target = "vitis_hls", mode = "csyn", project = "online_softmax.prj")()
    #mod = s2.build(target = "vitis_hls", mode = "csyn", project = "flash_attention.prj")()
    #rint("finished csynth")
    mod = s2.build(target = "vitis_hls", mode = "sw_emu", project = "sw_flash_attention.prj")(Q, K, V, my_solution)
    np.testing.assert_allclose(my_solution, solution, atol=1e-5)
    print("everything passed!")
    #mod = s3.build(target = "vitis_hls", mode = "csyn", project = "flash_attention.prj")()


if __name__ == "__main__":
    test_flash_attention()

