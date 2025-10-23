# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import torch
import torch.nn.functional as F
import allo
from allo.ir.types import bfloat16, Stream
import allo.dataflow as df
import numpy as np
from ml_dtypes import bfloat16 as np_bfloat16
from allo.memory import Layout
from allo.backend.aie.external_kernel import ExternalModule

np.random.seed(42)
os.environ["ENABLE_AGGRESSIVE_PORT_UTILIZATION_PATCH"] = "1"

# ===============================================================================
# Model Configuration
# ===============================================================================
N = 1024
D = 64

Q = np.random.randn(N, D) * 0.5
K = np.random.randn(N, D) * 0.5
V = np.random.randn(N, D) * 0.5


# ===============================================================================
# Torch Version
# ===============================================================================
def scaled_dot_product_attention(q, k, v):
    scores = torch.matmul(q, k.transpose(-2, -1))
    attn = F.softmax(scores, dim=-1)
    output = torch.matmul(attn, v)
    return output


# ===============================================================================
# Allo Version
# ===============================================================================
KERNEL_LIB_PATH = "../../../../allo/library/aie/"
Ty = bfloat16
softmax = ExternalModule(
    top="softmax_bf16",
    impl_path=KERNEL_LIB_PATH + "softmax_bf16.cc",
    input_idx=[0],
    output_idx=[1],
)


ATTN_P0 = N // 32
ATTN_P1 = N // 32
ATTN_SCORE_M_TILE = ATTN_P0 * 32
ATTN_SCORE_N_TILE = ATTN_P1 * 32
ATTN_SCORE_LyA = Layout("S0R")
ATTN_SCORE_LyB = Layout("S1R")
ATTN_SCORE_LyC = Layout("S0S1")


Mt, Nt = 64, 64
Pk, Pm, Pn = D // 64, N // 64, N // 64
LyA = Layout("S1S2")
LyB = Layout("S2S0")
LyC = Layout("S1S0")


def gen_attn_score_primitives():
    col_num = 4
    row_num = 4
    # chain on k dimension
    mapping_primitives = []
    bases: list[list[str]] = []
    for i in range(Pm):
        bases.append([])
        for j in range(Pn):
            base = f"gemm_0_{i}_{j}"
            for k in range(1, Pk):
                mapping_primitives.append(("chain", [base, f"gemm_{k}_{i}_{j}"]))
                base += f"-gemm_{k}_{i}_{j}"
            bases[i].append(base)

    if Pn // col_num < 1 or Pm // row_num < 1:
        col_num, row_num = row_num, col_num

    if Pn // col_num > 1 or Pm // row_num > 1:
        for i in range(row_num):
            for j in range(col_num):
                bundle_list = []
                for p in range(Pm // row_num):
                    for q in range(Pn // col_num):
                        bundle_list.append(bases[i + row_num * p][j + col_num * q])
                mapping_primitives.append(("bundle", bundle_list))

    return mapping_primitives


@df.region()
def attn_score_kernel():
    pipe: Stream[Ty[Mt, Nt], 2][Pk - 1, Pm, Pn]

    @df.kernel(mapping=[Pk, Pm, Pn])
    def gemm(A: Ty[N, D] @ LyA, B: Ty[D, N] @ LyB, C: Ty[N, N] @ LyC):
        pk, pm, pn = df.get_pid()
        C_in: Ty[Mt, Nt]
        with allo.meta_if(pk > 0):
            C_in[:, :] = pipe[pk - 1, pm, pn].get()
        with allo.meta_else():
            C_in[:, :] = 0
        C_out: Ty[Mt, Nt] = allo.add(allo.matmul(A, B), C_in)
        with allo.meta_if(pk < Pk - 1):
            pipe[pk, pm, pn].put(C_out)
        with allo.meta_elif(pk == Pk - 1):
            C[:, :] = C_out


attn_score_mod = df.build(
    attn_score_kernel,
    target="aie",
    project="attn_score.prj",
    mapping_primitives=gen_attn_score_primitives(),
    profile=True,
    warmup=200,
    num_iters=1000,
)


SOFTMAX_P0 = N // 4
SOFTMAX_Ly = Layout("S0R")


def gen_softmax_primitives():
    SOFTMAX_ROW = 16
    primitives = []
    for row in range(SOFTMAX_ROW):
        if SOFTMAX_P0 // SOFTMAX_ROW > 1:
            primitives.append(
                (
                    "bundle",
                    [
                        f"core_{SOFTMAX_ROW*i_+row}"
                        for i_ in range(SOFTMAX_P0 // SOFTMAX_ROW)
                    ],
                )
            )
    return primitives


@df.region()
def softmax_kernel():
    @df.kernel(mapping=[SOFTMAX_P0])
    def core(
        input_x: Ty[N, N] @ SOFTMAX_Ly,
        output_x: Ty[N, N] @ SOFTMAX_Ly,
    ):
        softmax(input_x, output_x)


softmax_mod = df.build(
    softmax_kernel,
    target="aie",
    project="softmax.prj",
    mapping_primitives=gen_softmax_primitives(),
    profile=True,
    warmup=200,
    num_iters=1000,
)

Mt, Nt = 64, 64
Pk, Pm, Pn = N // 64, N // 64, D // 64
LyA = Layout("S1S2")
LyB = Layout("S2S0")
LyC = Layout("S1S0")


@df.region()
def top():
    pipe: Stream[Ty[Mt, Nt], 2][Pk - 1, Pm, Pn]

    @df.kernel(mapping=[Pk, Pm, Pn])
    def gemm(A: Ty[N, N] @ LyA, B: Ty[N, D] @ LyB, C: Ty[N, D] @ LyC):
        pk, pm, pn = df.get_pid()
        C_in: Ty[Mt, Nt]
        with allo.meta_if(pk > 0):
            C_in[:, :] = pipe[pk - 1, pm, pn].get()
        with allo.meta_else():
            C_in[:, :] = 0
        C_out: Ty[Mt, Nt] = allo.add(allo.matmul(A, B), C_in)
        with allo.meta_if(pk < Pk - 1):
            pipe[pk, pm, pn].put(C_out)
        with allo.meta_elif(pk == Pk - 1):
            C[:, :] = C_out


def gen_gemm_primitive():
    ROW = 16
    # chain on k dimension
    mapping_primitives = []
    bases: list[list[str]] = []
    for i in range(Pm):
        bases.append([])
        for j in range(Pn):
            base = f"gemm_0_{i}_{j}"
            for k in range(1, Pk):
                mapping_primitives.append(("chain", [base, f"gemm_{k}_{i}_{j}"]))
                base += f"-gemm_{k}_{i}_{j}"
            bases[i].append(base)

    if Pm // ROW > 1:
        for i in range(ROW):
            bundle_list = []
            for p in range(Pm // ROW):
                bundle_list.append(bases[i + ROW * p][0])
            mapping_primitives.append(("bundle", bundle_list))

    return mapping_primitives


gemm2_mod = df.build(
    top,
    project="gemm.prj",
    target="aie",
    mapping_primitives=gen_gemm_primitive(),
    profile=True,
    warmup=200,
    num_iters=1000,
)

# allo output
Q_ = Q.astype(np_bfloat16)
K_ = K.astype(np_bfloat16)
V_ = V.astype(np_bfloat16)

attention_score = np.empty((N, N), dtype=np_bfloat16)
attn_score_mod(Q_, K_.T, attention_score)
attn_weight = np.zeros((N, N)).astype(np_bfloat16)
softmax_mod(attention_score, attn_weight)
x = np.zeros((N, D)).astype(np_bfloat16)
gemm2_mod(attn_weight, V_, x)

# sample output
q = torch.from_numpy(Q).to(dtype=torch.bfloat16)
k = torch.from_numpy(K).to(dtype=torch.bfloat16)
v = torch.from_numpy(V).to(dtype=torch.bfloat16)
output = scaled_dot_product_attention(q, k, v)

# compare
np.testing.assert_allclose(
    x.astype(np.float32), output.to(torch.float32).numpy(), atol=1e-1
)
print("Allo bfloat16 attn matches PyTorch bfloat16 reference within tolerance ✔️")
del os.environ["ENABLE_AGGRESSIVE_PORT_UTILIZATION_PATCH"]
