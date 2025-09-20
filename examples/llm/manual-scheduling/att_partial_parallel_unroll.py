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
from allo.ir.transform import find_buffer

# --------------------------------------------------------------------------------
# Problem size: 1 head of 32 × 32 with 64-wide model dimension
# (easy to synthesize quickly; raise L/D/H for bigger experiments)
# --------------------------------------------------------------------------------
H, L, D = 8, 64, 1024
P = H // 4 # parallel heads
h_d:int32 = D // H
Ty = float32
MIN_FLOAT32:Ty = -3.402823466e+38  # minimum float32 value



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

def gemm_2(A: float32[L, L], B: float32[L, D], start_pos: int32) -> float32[L, h_d]:
    C: float32[L, h_d] = 0.0
    for i in range(L):
        for j in range(h_d):
            for k in allo.reduction(L):
                C[i, j] += A[i, k] * B[k, start_pos + j]
    return C

def custom_softmax(QK_in: Ty[L, L]) -> Ty[L, L]:
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

# ------------------------------------------------------------------------------
def attention_parallel_subset(Q: Ty[L, D], K: Ty[L, D], V: Ty[L, D])->Ty[L, P*h_d]:
    Z_i: Ty[L, P*h_d]
    start_pos:int32 = 0
    for j in allo.grid(P, name="multi_head_inner_loop"):
        start_pos = j * h_d
        QK_t = gemm(Q, K, start_pos)
        QK_t_s = custom_softmax(QK_t)
        Z_i_j = gemm_2(QK_t_s, V, start_pos)
        for (i2, j2) in allo.grid(L, h_d, name="store_intermediate_result"):
            Z_i[i2, start_pos + j2] = Z_i_j[i2, j2]
    return Z_i


def attention_parallel_full(Q: Ty[L, D], K: Ty[L, D], V: Ty[L, D]) -> Ty[L, D]:
    Z: Ty[L, D]
    for i in allo.grid(H//P, name="multi_head_outer_loop"):
        Z_new = attention_parallel_subset(Q, K, V)
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
    # solution = sdp(Q, K, V, H, D)
    
    s1 = allo.customize(gemm)
    s2 = allo.customize(gemm_2)
    s3 = allo.customize(custom_softmax)
    s5 = allo.customize(attention_parallel_subset)

    s1.reorder("k", "j")
    s1.buffer_at(s1.C, "i")
    s1.pipeline("k")
    s1.unroll(axis="j", factor=H)
    s5.compose([s1, s2, s3])
    s5.unroll(axis="j", factor=P)

    s6 = allo.customize(attention_parallel_full)

    s6.partition(s6.Q, partition.Block, dim=2, factor=H)
    s6.partition(s6.K, partition.Block, dim=2, factor=H)
    s6.partition(s6.V, partition.Block, dim=2, factor=H)
    s6.partition(s6.Z, partition.Block, dim=2, factor=H)
    
    s6.compose([s5])
    print(s6.get_loops("attention_parallel_full")["multi_head_outer_loop"])
    s6.dataflow("attention_parallel_subset")
    s6.dataflow("attention_parallel_full")
    # s6.dataflow(s6.get_loops("attention_parallel_full")["multi_head_outer_loop"]["i2"])
    # print(s5.module)
    # print(s5.get_loops("attention_parallel_subset"))
    
    # print(s6.module)
    # assert False, "Passed test_attention!"

    # print(s5.module)
    s6.build(target="vitis_hls", mode="csyn", project=f"att_unroll_par_{P}_heads_{H}_no_df.prj")()
    print("everything passed!")
if __name__ == "__main__":
    test_attention()