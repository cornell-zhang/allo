# test_attention_autoscheduler.py
#
# Three builds:
#   A. naive_baseline.prj      – no scheduler, no systolic array
#   B. naive_autosched.prj     – combined scheduler on naïve kernel
#   C. systolic_autosched.prj  – author's systolic kernel + graph scheduler
#
# Run with:  pytest -q test_attention_autoscheduler.py
#
# --------------------------------------------------------------------
import math, pytest, allo
import numpy as np
from allo._mlir import ir as mlir_ir
StringAttr = mlir_ir.StringAttr
from allo.ir.types import float32, int32
from allo.autoscheduler.passes import dataflow_optimization_pass
from allo.autoscheduler.config import AutoschedulerConfig
from allo.autoscheduler.dfg import DFG
from gurobipy import GurobiError


# --------------------------------------------------------------------------------
# Problem size: 1 head of 32 × 32 with 64-wide model dimension
# (easy to synthesize quickly; raise L/D/H for bigger experiments)
# --------------------------------------------------------------------------------
H, L, D = 1, 8, 8
h_d:int32 = D // H
Ty = float32                                   # shorthand
MIN_FLOAT32:Ty = -3.402823466e+38  # minimum float32 value





# ------------------------------------------------------------------------------
# Helper: give MLIR block arguments names so Allo's HLS wrapper pass is happy
# ------------------------------------------------------------------------------
def _name_args(schedule):
    blk = schedule.module.operation.regions[0].blocks[0]
    for idx, arg in enumerate(blk.arguments):
        arg.owner.attributes["name"] = StringAttr.get(f"arg{idx}",
                                                      schedule.module.context)

# ------------------------------------------------------------------------------
# Naïve scaled-dot-product attention (single head, no systolic array, no tiling)
# ------------------------------------------------------------------------------
import math
import numpy as np
float32_scalar = float32        

def gemm(A: float32[L, h_d], B: float32[h_d, L]) -> float32[L, L]:
    C: float32[L, L]
    for i in range(L):
        for j in range(L):
            for k in allo.reduction(h_d):
                C[i, j] += A[i, k] * B[k, j]
    return C

def gemm_2(A: float32[L, L], B: float32[L, h_d]) -> float32[L, h_d]:
    C: float32[L, h_d]
    for i in range(L):
        for j in range(h_d):
            for k in allo.reduction(L):
                C[i, j] += A[i, k] * B[k, j]
    return C    


#Slice helper function
def slice_3_matrixes(Q: Ty[L, D], K: Ty[L, D], V: Ty[L, D], start_pos: int32, h_d: int32) -> (Ty[L, h_d], Ty[L, h_d], Ty[L, h_d]):
    # Q: Ty[L, D]
    # start_pos: scalar 
    outQ: Ty[L, h_d] = 0.0
    outK: Ty[L, h_d] = 0.0
    outV: Ty[L, h_d] = 0.0
    for i, j in allo.grid(L, h_d):               # i in [0,L), j in [0,h_d)
        # copy Q[i, start_pos + j] into out[i, j]
        outQ[i, j] = Q[i, start_pos + j]
        outK[i, j] = K[i, start_pos + j]
        outV[i, j] = V[i, start_pos + j]
    return (outQ, outK, outV)

def transpose_matrix(M: Ty[L, h_d]) -> Ty[h_d, L]:
    out: Ty[h_d, L]
    for ii, jj in allo.grid(L, h_d):               
        out[jj, ii] = M[ii, jj]
    return out

def custom_softmax(QK_in: Ty[L, L]) -> Ty[L, L]:
    exp_buf:Ty[L, L]      # TEMP for exponentials
    QK_out: Ty[L, L]
    max_vals: Ty[L] = MIN_FLOAT32
    rows_total: Ty[L]
    invs: Ty[L]
    # --- Kernel B: scan QK_in and update max_vals in a perfect 2D nest ------
    for ii in range(L):
        for jj in range(L):
            v:Ty = QK_in[ii, jj]
            m:Ty = max_vals[ii]                   # affine.load
            # this select lowers to arithmetic, not a branch
            max_vals[ii] = v if v > m else m   # affine.store

    # --- Kernel C: compute exponentials and row sums (another perfect nest) -
    for ii in range(L):
        # rows_total[i] = 0.0
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

def get_attention_small_dimension(Q: Ty[L, h_d],
                                  K: Ty[L, h_d],
                                  V: Ty[L, h_d]) -> Ty[L, h_d]:
    K_h_t = transpose_matrix(K)  # Kᵀ : (L × h_d) → (h_d × L)
    QK_t = gemm(Q, K_h_t) # QKᵀ : (L × h_d) · (h_d × L) → (L × L)
    QK_t_s = custom_softmax(QK_t)  # softmax over rows
    # # SV : (L × L) · (L × D) → (L × D)
    Z = gemm_2(QK_t_s, V)  # (L × L) · (L × h_d) → (L × h_d)
    return Z
    
def get_matrix_slice(A:Ty[L, D], start_pos: int32) -> Ty[L, h_d]:
    B: Ty[L, h_d] = 0.0
    for ii, jj in allo.grid(L, h_d):
        B[ii, jj] = A[ii, start_pos + jj]
    return B

def write_slice_to_matrix(Z: Ty[L, D], Z_h: Ty[L, h_d], start_pos: int32):
    for ii, jj in allo.grid(L, h_d):
        Z[ii, start_pos + jj] = Z_h[ii, jj]



# ------------------------------------------------------------------------------
def attention(Q: Ty[L, D], K: Ty[L, D], V: Ty[L, D]) -> Ty[L, D]:
    Z: Ty[L, D] = 0.0
    for i in allo.grid(H):
        start_pos:int32 = i * h_d
        # Q_h = get_matrix_slice(Q, start_pos)  # Q_h : (L × h_d)
        # K_h = get_matrix_slice(K, start_pos)  # K_h : (L × h_d)
        # V_h = get_matrix_slice(V, start_pos)  # V_h : (L × h_d)
        Q_h:Ty[L, h_d] = 0.0
        K_h:Ty[L, h_d] = 0.0
        V_h:Ty[L, h_d] = 0.0
        for i2, j in allo.grid(L, h_d):
            Q_h[i2, j] = Q[i2, start_pos + j]
            K_h[i2, j] = K[i2, start_pos + j]
            V_h[i2, j] = V[i2, start_pos + j]
        #now we have the values to be passed into get_attention_small_dimension
        K_h_t = transpose_matrix(K)  # Kᵀ : (L × h_d) → (h_d × L)
        QK_t = gemm(Q, K_h_t) # QKᵀ : (L × h_d) · (h_d × L) → (L × L)
        QK_t_s = custom_softmax(QK_t)  # softmax over rows
        # # SV : (L × L) · (L × D) → (L × D)
        Z_h = gemm_2(QK_t_s, V)  # (L × L) · (L × h_d) → (L × h_d)
        # write_slice_to_matrix(Z, Z_h, start_pos)
        for ii, j in allo.grid(L, h_d):
            Z[ii, start_pos + j] = Z_h[ii, j]
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

from allo.library.nn import scaled_dot_product_attention as allo_sdp
# ------------------------------------------------------------------------------
# PyTest top-level experiment
# ------------------------------------------------------------------------------
def test_attention():
    # optimized_sdp = allo.customize(allo_sdp, instantiate=[float32, H, L, D, 2, 2])
    # print(optimized_sdp.module)
    # mod = optimized_sdp.build(target="vitis_hls",
    #                           mode="csyn",
    #                           project="naive_baseline.prj")()
    # s0 = allo.customize(get_matrix_slice)
    # s0 = dataflow_optimization_pass(s0, kind="graph", verbose=True)
    
    # Create autoscheduler config
    cfg = AutoschedulerConfig.builder().with_kind("graph").with_verbose(True)
    
    s1 = allo.customize(gemm)
    s1 = dataflow_optimization_pass(s1, cfg)
    # s1.inline('affine_kernel')
    print(s1.module)
    s2 = allo.customize(gemm_2)
    s2 = dataflow_optimization_pass(s2, cfg)
    print(s2.module)
    s3 = allo.customize(custom_softmax)
    s3 = dataflow_optimization_pass(s3, cfg)
    s4 = allo.customize(transpose_matrix)
    s4 = dataflow_optimization_pass(s4, cfg)
    # s5 = allo.customize(write_slice_to_matrix)
    # s5 = dataflow_optimization_pass(s5, kind="graph", verbose=True)

    # s5 = allo.customize(get_attention_small_dimension)
    a = allo.customize(attention)
    # a.unroll(axis="i", factor=H)  # unroll the outer loop
    a.compose([s1, s2, s3, s4])
    # a.inline('gemm')
    # a.inline('gemm_2')
    # a.inline('custom_softmax')
    # a.inline('transpose_matrix')
    print(a.module)
    # a.unroll(axis="i", factor=H)  # unroll the outer loop
    # print(a.module)
    # s5.compose([s1, s2, s3, s4])
    # s0.inline()
    # s1.inline()
    # s2.inline()
    # s3.inline()
    # s4.inline()
    # s5.inline()
    a = dataflow_optimization_pass(a, cfg)
    # a.compose([s0, s1, s2, s3, s4, s5])
    # a.inline('get_matrix_slice')
    # a.inline('write_slice_to_matrix')
    # a.inline('transpose_matrix')
    # a.inline('gemm')
    # a.inline('gemm_2')
    # a.inline('custom_softmax')
    # print(a.module)
    # a = dataflow_optimization_pass(a, kind="graph", verbose=True)
    # assert False, "This test is not ready yet, please use the small_attention baseline instead"
    mod = a.build(target="vitis_hls",
             mode="csyn",
             project="small_attention_baseline.prj")()
    Q = np.random.rand(L, D).astype(np.float32)
    K = np.random.rand(L, D).astype(np.float32)
    V = np.random.rand(L, D).astype(np.float32)
    Z = np.random.rand(L, h_d).astype(np.float32)
    # solution = sdp(Q, K, V, H, D)
    print("starting baseline run")
    # mod(Q, K, V, Z)
    # assert np.allclose(Z, solution, rtol=1e-5, atol=1e-5), "Results do not match!"
    print("finished baseline run")
    s_gemm  = allo.customize(gemm)
    s_gemm2 = allo.customize(gemm_2)
    s_soft = allo.customize(custom_softmax)
    s_trans = allo.customize(transpose_matrix)
    small_attention = allo.customize(get_attention_small_dimension)
    small_attention.compose([s_gemm, s_gemm2, s_soft, s_trans])
    small_attention = dataflow_optimization_pass(small_attention, cfg)
    print("finished optimization pass")
    mod2 = small_attention.build(target="vitis_hls",
                          mode="csyn",
                            project="small_attention_autosched.prj")()
    # Z2 = np.random.rand(L, h_d).astype(np.float32)
    print("starting optimized run")
    # mod2(Q, K, V, Z2)
    print("finished optimized run")
    # assert np.allclose(Z2, solution, rtol=1e-5, atol=1e-5), "Results do not match!"
    # print(Z)
    print("-"* 80)
    # print(solution)
    print("-"* 80)
    # print(Z2)
if __name__ == "__main__":
    test_attention()
    print("All tests passed!")
