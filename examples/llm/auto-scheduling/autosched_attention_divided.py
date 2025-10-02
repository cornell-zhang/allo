# --------------------------------------------------------------------
#  Option B: schedule each leaf kernel with the autoscheduler, then
#  compose them into the original naïve data-flow graph.
# --------------------------------------------------------------------
import math, pytest, allo                           # Allo front-end
from allo.ir.types import float32, int32            # MLIR types
from allo.autoscheduler.passes import dataflow_optimization_pass
from allo.autoscheduler.dfg    import DFG
from gurobipy import GurobiError                    # only needed if you
                                                    # call dfg .model()
# --------------------------------------------------------------------
# Problem size (single head of a 32×32 sequence, model width 64)
H, L, D  = 4, 32, 64
h_d      = D // H
Ty       = float32              # shorthand alias for Allo type hints
# --------------------------------------------------------------------
# --- leaf perfect-affine kernels -----------------------------------
def gemm(A: Ty[L, h_d], B: Ty[h_d, L]) -> Ty[L, L]:
    C: Ty[L, L] = 0.0
    for i, j in allo.grid(L, L):
        for k in allo.reduction(h_d):
            C[i, j] += A[i, k] * B[k, j]
    return C

def gemm_2(A: Ty[L, L], B: Ty[L, h_d]) -> Ty[L, h_d]:
    C: Ty[L, h_d] = 0.0
    for i, j in allo.grid(L, h_d):
        for k in allo.reduction(L):
            C[i, j] += A[i, k] * B[k, j]
    return C

def transpose_matrix(M: Ty[L, h_d]) -> Ty[h_d, L]:
    out: Ty[h_d, L] = 0.0
    for i, j in allo.grid(L, h_d):
        out[j, i] = M[i, j]
    return out

def row_max(QK: Ty[L, L]) -> Ty[L]:
    M: Ty[L] = 0.0
    for i, j in allo.grid(L, L):
        if j == 0 or QK[i, j] > M[i]:
            M[i] = QK[i, j]
    return M

def row_max(QK: Ty[L, L]) -> Ty[L]:
    M : Ty[L] = 0.0
    for i in allo.grid(L):
        max_val: Ty = QK[i, 0]
        for j in range(1, L):          # simple python range is OK
            if QK[i, j] > max_val:
                max_val = QK[i, j]
        M[i] = max_val
    return M
# --------------------------------------------------------------------
# --- top naïve data-flow kernel ------------------------------------
def attention_naive(Q: Ty[L, D], K: Ty[L, D], V: Ty[L, D]) -> Ty[L, D]:
    Z: Ty[L, D] = 0.0
    for head in range(H):
        offset: int32 = head * h_d
        # slice Q/K/V to the current head
        Qh: Ty[L, h_d] = 0.0; Kh: Ty[L, h_d] = 0.0; Vh: Ty[L, h_d] = 0.0
        for i, j in allo.grid(L, h_d):
            idx: int32 = offset + j
            Qh[i, j] = Q[i, idx]
            Kh[i, j] = K[i, idx]
            Vh[i, j] = V[i, idx]

        Kht  = transpose_matrix(Kh)
        QK   = gemm(Qh, Kht)
        M    = row_max(QK)
        QKs  = row_softmax(QK, M)
        SV   = gemm_2(QKs, Vh)

        for i, j in allo.grid(L, h_d):
            Z[i, offset + j] = SV[i, j]
    return Z

# ----------------------------------------
# leaf  ❱❱  soft-max over one row
# ----------------------------------------
def row_softmax(QK: Ty[L, L]) -> Ty[L, L]:
    S    : Ty[L, L] = 0.0
    maxv : Ty[L]    = 0.0
    rows : Ty[L]    = 0.0

    # ❶ max
    for i, j in allo.grid(L, L):
        if j == 0 or QK[i, j] > maxv[i]:
            maxv[i] = QK[i, j]

    # ❷ exp + running row-sum
    for i, j in allo.grid(L, L):
        S[i, j] = allo.exp(QK[i, j] - maxv[i])
        rows[i] += S[i, j]

    # ❸ normalise
    for i, j in allo.grid(L, L):
        S[i, j] = S[i, j] / rows[i]            # rows[i] is never 0 for soft-max

    return S

# --------------------------------------------------------------------
# ----------------------- test-bench --------------------------------
def test_attention():
    # schedule the leaves
    s_gemm  = dataflow_optimization_pass(allo.customize(gemm),           kind="graph")
    s_gemm2 = dataflow_optimization_pass(allo.customize(gemm_2),         kind="graph")
    s_tran  = dataflow_optimization_pass(allo.customize(transpose_matrix),kind="graph")
    # s_rmax  = dataflow_optimization_pass(allo.customize(row_max),        kind="graph")
    s_rmax  = allo.customize(row_max)  
    # s_soft  = dataflow_optimization_pass(allo.customize(row_softmax),    kind="graph")
    s_soft  = allo.customize(row_softmax)

    # compose them back under the naïve producer-consumer graph
    top = allo.customize(attention_naive)
    top.compose([s_gemm, s_gemm2, s_tran, s_rmax, s_soft])

    # build (csyn = C-synthesis only, quick)
    top.build(target="vitis_hls", mode="csyn", project="naive_autosched.prj")()

if __name__ == "__main__":
    pytest.main([__file__])
