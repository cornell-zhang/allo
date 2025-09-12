# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Model Architecture: https://huggingface.co/openai-community/gpt2
    ```
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    print(model)
    ```

    GPT2LMHeadModel(
        (transformer): GPT2Model(
            (wte): Embedding(50257, 768)
            (wpe): Embedding(1024, 768)
            (drop): Dropout(p=0.1, inplace=False)
            (h): ModuleList(
                ####################################################################
                (0-11): 12 x GPT2Block(
                    (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                    (attn): GPT2Attention(
                        (c_attn): Conv1D(nf=2304, nx=768) # nf = 3 * 768
                        (c_proj): Conv1D(nf=768, nx=768)
                        (attn_dropout): Dropout(p=0.1, inplace=False)
                        (resid_dropout): Dropout(p=0.1, inplace=False)
                    )
                    (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                    (mlp): GPT2MLP(
                        (c_fc): Conv1D(nf=3072, nx=768)
                        (c_proj): Conv1D(nf=768, nx=3072)
                        (act): NewGELUActivation()
                        (dropout): Dropout(p=0.1, inplace=False)
                    )
                )
                ####################################################################
            )
            (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        )
        (lm_head): Linear(in_features=768, out_features=50257, bias=False)
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import allo
import allo.dataflow as df
from allo.ir.types import float32, bfloat16, int32
from allo.memory import Layout
from allo.backend.aie import ExternalModule

torch.manual_seed(0)
np.random.seed(0)

# ===============================================================================
# Model Configuration
# ===============================================================================
USE_ALL_NPU_KERNELS = True  # if False, we will offload softmax and gelu to cpu
KERNEL_LIB_PATH = "../../../../allo/library/aie/"
BATCH = 1  # fixme: don't care for now
SEQ = 64
EMBD = 768  # 64 * 12
N_HEAD = 12
HEAD_DIM = EMBD // N_HEAD
FFN_HID = EMBD * 4

assert SEQ == 64, "SEQ must be 64 (to use masked softmax external kernel)"
assert EMBD % 64 == 0, "EMBD must be a multiple of 64"
assert HEAD_DIM % 64 == 0, "HEAD_DIM must be a multiple of 64"


# ===============================================================================
# Torch Version
# ===============================================================================
class MiniGPT2(nn.Module):
    """
    References
        - GPT2Block forward: https://github.com/huggingface/transformers/blob/2166b6b4ff09f6dd3867ab982f262f66482aa968/src/transformers/models/gpt2/modeling_gpt2.py#L388
    """

    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(EMBD, N_HEAD, batch_first=True)
        self.ln_1 = nn.LayerNorm(EMBD, elementwise_affine=True)
        self.ffn_up = nn.Linear(EMBD, FFN_HID, bias=False)
        self.ffn_down = nn.Linear(FFN_HID, EMBD, bias=False)
        self.gelu = nn.GELU()
        self.ln_2 = nn.LayerNorm(EMBD, elementwise_affine=True)
        self.attn.in_proj_bias.data.zero_()
        self.attn.out_proj.bias.data.zero_()

    def get_sample_attn_(self, x):
        def scaled_dot_product_attention(q, k, v, attn_mask=None):
            d_k = q.size(-1)
            scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(
                torch.tensor(d_k, dtype=torch.float32)
            )
            if attn_mask is not None:
                scores = scores.masked_fill(attn_mask == 0, float("-inf"))

            attn = F.softmax(scores, dim=-1)
            output = torch.matmul(attn, v)
            return output

        def manual_multihead_attention(x):
            Wqkv = self.attn.in_proj_weight
            Wq, Wk, Wv = torch.split(Wqkv, EMBD, dim=0)
            Wo = self.attn.out_proj.weight
            # Broadcast mask
            mask = ~torch.triu(torch.ones(SEQ, SEQ), 1).bool()
            mask.unsqueeze(0)
            # Linear projections
            q = x @ Wq.T
            k = x @ Wk.T
            v = x @ Wv.T

            # Reshape to (SEQ, N_HEAD, HEAD_DIM)
            q = q.view(SEQ, N_HEAD, HEAD_DIM).transpose(0, 1)
            k = k.view(SEQ, N_HEAD, HEAD_DIM).transpose(0, 1)
            v = v.view(SEQ, N_HEAD, HEAD_DIM).transpose(0, 1)

            # Compute attention
            out = scaled_dot_product_attention(q, k, v, mask)
            # Concatenate heads
            out = out.transpose(0, 1).contiguous().view(SEQ, EMBD)
            # Final linear projection
            out = out @ Wo.T
            return out

        return manual_multihead_attention(x)

    def forward(self, x: torch.Tensor):
        residual = x
        x = self.ln_1(x)
        attn_out, _ = self.attn(
            x,
            x,
            x,
            need_weights=False,
            attn_mask=torch.triu(torch.ones(SEQ, SEQ), 1).bool(),
        )
        x = attn_out + residual
        residual = x
        x = self.ln_2(x)
        activeated_x = self.gelu(self.ffn_up(x))
        x = self.ffn_down(activeated_x)
        x = residual + x
        return x


# ===============================================================================
# Allo Version
# ===============================================================================
Ty = float32  # All tensors use float32
N = BATCH * SEQ  # 16   flattened (batch*seq)


def run(x_fp32: np.ndarray, params: dict):

    # ----------------------------------------------------------------
    # LayerNorm
    # ----------------------------------------------------------------
    norm = ExternalModule(
        top="layer_norm",
        impl_path=KERNEL_LIB_PATH + "layer_norm.cc",
        input_idx=[0, 1],
        output_idx=[2],
    )
    NORM_P0 = 4
    NORM_SEQ_TILE = 16
    NORM_TILE = NORM_SEQ_TILE // NORM_P0
    norm_io_layout = Layout("S0R")
    norm_arg_layout = Layout("R")

    @df.region()
    def layer_norm_kernel():
        pipe = df.array(
            df.pipe(dtype=Ty, shape=(NORM_TILE, EMBD), depth=1), shape=(NORM_P0,)
        )

        @df.kernel(mapping=[NORM_P0])
        def norm_no_bias(
            input_x: Ty[NORM_SEQ_TILE, EMBD] @ norm_io_layout,
            weight: Ty[EMBD] @ norm_arg_layout,
        ):
            pi = df.get_pid()
            tmp: Ty[NORM_TILE, EMBD] = 0
            norm(input_x, weight, tmp)
            pipe[pi].put(tmp)

        @df.kernel(mapping=[NORM_P0])
        def norm_add_bias(
            bias: Ty[EMBD] @ norm_arg_layout,
            output_x: Ty[NORM_SEQ_TILE, EMBD] @ norm_io_layout,
        ):
            pi = df.get_pid()
            data = pipe[pi].get()
            output_x[:, :] = allo.add(data, bias)

    # ----------------------------------------------------------------
    # Linear
    # ----------------------------------------------------------------
    LINEAR_M, LINEAR_N, LINEAR_K = 64, 64, 64
    linear_A_layout = Layout("S0R")
    linear_B_layout = Layout("RS1")
    linear_C_layout = Layout("S0S1")

    @df.region()
    def linear_matmul_kernel():
        @df.kernel(mapping=[4, 4])
        def gemm(
            A: Ty[LINEAR_M, LINEAR_K] @ linear_A_layout,
            B: Ty[LINEAR_K, LINEAR_N] @ linear_B_layout,
            C: Ty[LINEAR_M, LINEAR_N] @ linear_C_layout,
        ):
            C[:, :] = allo.matmul(A, B)

    @df.region()
    def linear_accumulate_kernel():
        @df.kernel(mapping=[2, 4])
        def core(
            A: Ty[LINEAR_M, LINEAR_N] @ linear_C_layout,
            B: Ty[LINEAR_M, LINEAR_N] @ linear_C_layout,
            C: Ty[LINEAR_M, LINEAR_N] @ linear_C_layout,
        ):
            C[:, :] = allo.add(A, B)

    # ----------------------------------------------------------------
    # Attention Score
    # ----------------------------------------------------------------
    attn_score = ExternalModule(
        top="transpose_matmul_with_scale",
        impl_path=KERNEL_LIB_PATH + "transpose_matmul_with_scale.cc",
        input_idx=[0, 1],
        output_idx=[2],
    )
    ATTN_P0 = 2
    ATTN_P1 = 2
    ATTN_SCORE_M_TILE = ATTN_P0 * 32
    ATTN_SCORE_N_TILE = ATTN_P1 * 32
    ATTN_SCORE_LyA = Layout("S0R")
    ATTN_SCORE_LyB = Layout("S1R")
    ATTN_SCORE_LyC = Layout("S0S1")

    @df.region()
    def attn_score_kernel():
        @df.kernel(mapping=[ATTN_P0, ATTN_P1])
        def core(
            A: Ty[ATTN_SCORE_M_TILE, HEAD_DIM] @ ATTN_SCORE_LyA,
            B: Ty[ATTN_SCORE_N_TILE, HEAD_DIM] @ ATTN_SCORE_LyB,
            C: Ty[ATTN_SCORE_M_TILE, ATTN_SCORE_N_TILE] @ ATTN_SCORE_LyC,
        ):
            attn_score(A, B, C)

    # ----------------------------------------------------------------
    # Masked Softmax
    # ----------------------------------------------------------------
    masked_softmax = ExternalModule(
        top="masked_softmax_float32",
        impl_path=KERNEL_LIB_PATH + "masked_softmax.cc",
        input_idx=[0, 1],
        output_idx=[2],
    )
    Tint = int32
    SOFTMAX_P0 = 2
    SOFTMAX_P1 = 3
    SOFTMAX_HEAD_TILE = SOFTMAX_P1
    SOFTMAX_SEQ_TILE = SEQ // SOFTMAX_P0
    SOFTMAX_Ly = Layout("S1S0")
    SOFTMAX_ROW_Ly = Layout("S1")

    @df.region()
    def masked_softmax_kernel():
        @df.kernel(mapping=[SOFTMAX_P0, SOFTMAX_P1])
        def core(
            input_x: Ty[SEQ, SEQ * SOFTMAX_HEAD_TILE] @ SOFTMAX_Ly,
            row: Tint[SOFTMAX_P0] @ SOFTMAX_ROW_Ly,
            output_x: Ty[SEQ, SEQ * SOFTMAX_HEAD_TILE] @ SOFTMAX_Ly,
        ):
            masked_softmax(input_x, row, output_x)

    # ----------------------------------------------------------------
    # Gelu
    # ----------------------------------------------------------------
    gelu = ExternalModule(
        top="gelu_float32",
        impl_path=KERNEL_LIB_PATH + "gelu.cc",
        input_idx=[0],
        output_idx=[1],
    )
    GELU_P0 = 4
    GELU_P1 = 4
    GELU_SEQ_TILE = 16
    GELU_Ly = Layout("S0S1")

    @df.region()
    def gelu_kernel():
        @df.kernel(mapping=[GELU_P0, GELU_P1])
        def core(
            input_x: Ty[GELU_SEQ_TILE, FFN_HID] @ GELU_Ly,
            output_x: Ty[GELU_SEQ_TILE, FFN_HID] @ GELU_Ly,
        ):
            gelu(input_x, output_x)

    # ##############################################################
    # BUILD
    # ##############################################################
    layer_norm_mod = df.build(layer_norm_kernel, target="aie", project="norm.prj")
    linear_matmul_mod = df.build(
        linear_matmul_kernel, target="aie", project="linear_matmul.prj"
    )
    linear_accumulate_mod = df.build(
        linear_accumulate_kernel, target="aie", project="linear_accumulate.prj"
    )
    attn_score_mod = df.build(attn_score_kernel, target="aie", project="attn_score.prj")
    masked_softmax_mod = df.build(
        masked_softmax_kernel, target="aie", project="masked_softmax.prj"
    )
    gelu_mod = df.build(gelu_kernel, target="aie", project="gelu.prj")

    # ##############################################################
    # TOOL
    # ##############################################################
    def layernorm(input_x, weight, bias, output_x):
        for i in range(SEQ // NORM_SEQ_TILE):
            tile_input = input_x[i * NORM_SEQ_TILE : (i + 1) * NORM_SEQ_TILE, :]
            layer_norm_mod(
                tile_input,
                weight,
                bias,
                output_x[i * NORM_SEQ_TILE : (i + 1) * NORM_SEQ_TILE, :],
            )

    def linear_projection(A, B, C, M, N, K):
        for i in range(M // LINEAR_M):
            for j in range(N // LINEAR_N):
                C_tmp = np.zeros((LINEAR_M, LINEAR_N)).astype(np.float32)
                for k in range(K // LINEAR_K):
                    tile_A = A[
                        i * LINEAR_M : (i + 1) * LINEAR_M,
                        k * LINEAR_K : (k + 1) * LINEAR_K,
                    ]
                    tile_B = B[
                        k * LINEAR_K : (k + 1) * LINEAR_K,
                        j * LINEAR_N : (j + 1) * LINEAR_N,
                    ]
                    linear_matmul_mod(tile_A, tile_B, C_tmp)
                    linear_accumulate_mod(
                        C[
                            i * LINEAR_M : (i + 1) * LINEAR_M,
                            j * LINEAR_N : (j + 1) * LINEAR_N,
                        ],
                        C_tmp,
                        C[
                            i * LINEAR_M : (i + 1) * LINEAR_M,
                            j * LINEAR_N : (j + 1) * LINEAR_N,
                        ],
                    )

    def add_residual(residual, x, M, N):
        """
        reuse 'linear_accumulate_mod' for residual
        residual = residual + x
        """
        for i in range(M // LINEAR_M):
            for j in range(N // LINEAR_N):
                linear_accumulate_mod(
                    residual[
                        i * LINEAR_M : (i + 1) * LINEAR_M,
                        j * LINEAR_N : (j + 1) * LINEAR_N,
                    ],
                    x[
                        i * LINEAR_M : (i + 1) * LINEAR_M,
                        j * LINEAR_N : (j + 1) * LINEAR_N,
                    ],
                    residual[
                        i * LINEAR_M : (i + 1) * LINEAR_M,
                        j * LINEAR_N : (j + 1) * LINEAR_N,
                    ],
                )

    def masked_softmax(attention_score, attention_weight):
        row_idx = np.array(list(range(0, SEQ, SOFTMAX_SEQ_TILE))).astype(np.int32)
        for i in range(N_HEAD // SOFTMAX_HEAD_TILE):
            masked_softmax_mod(
                attention_score[
                    :, i * SOFTMAX_HEAD_TILE : (i + 1) * SOFTMAX_HEAD_TILE, :
                ],
                row_idx,
                attention_weight[
                    :,
                    i * (SOFTMAX_HEAD_TILE * SEQ) : (i + 1) * (SOFTMAX_HEAD_TILE * SEQ),
                ],
            )

    # ##############################################################
    # FORWARD
    # ##############################################################
    x = x_fp32.astype(np.float32)
    residual = x.reshape(SEQ, EMBD)
    x = np.empty((SEQ, EMBD), dtype=np.float32)
    layernorm(residual, params["W_norm_1"], params["b_norm_1"], x)

    # qkv projections (M = SEQ, N = EMBD, K = EMBD)
    query = np.zeros((SEQ, EMBD)).astype(np.float32)
    key = np.zeros((SEQ, EMBD)).astype(np.float32)
    value = np.zeros((SEQ, EMBD)).astype(np.float32)
    linear_projection(x, params["Wq"], query, SEQ, EMBD, EMBD)
    linear_projection(x, params["Wk"], key, SEQ, EMBD, EMBD)
    linear_projection(x, params["Wv"], value, SEQ, EMBD, EMBD)

    # attention score
    attention_score = np.empty((SEQ, N_HEAD, SEQ), dtype=np.float32)
    for i in range(SEQ // ATTN_SCORE_M_TILE):
        for j in range(SEQ // ATTN_SCORE_N_TILE):
            for k in range(N_HEAD):
                attn_score_mod(
                    query[
                        i * ATTN_SCORE_M_TILE : (i + 1) * ATTN_SCORE_M_TILE,
                        k * HEAD_DIM : (k + 1) * HEAD_DIM,
                    ],
                    key[
                        j * ATTN_SCORE_N_TILE : (j + 1) * ATTN_SCORE_N_TILE,
                        k * HEAD_DIM : (k + 1) * HEAD_DIM,
                    ],
                    attention_score[
                        i * ATTN_SCORE_M_TILE : (i + 1) * ATTN_SCORE_M_TILE,
                        k,
                        j * ATTN_SCORE_N_TILE : (j + 1) * ATTN_SCORE_N_TILE,
                    ],
                )

    # safe softmax
    if USE_ALL_NPU_KERNELS:
        attn_weight = np.zeros((SEQ, N_HEAD * SEQ)).astype(np.float32)
        masked_softmax(attention_score, attn_weight)
    else:
        mask = torch.triu(torch.ones(SEQ, SEQ), 1).bool()
        mask = np.repeat(mask[:, np.newaxis, :], N_HEAD, axis=1)
        attention_score[mask == 1] = -np.inf
        tensor_atten_score = torch.from_numpy(attention_score)
        attn_weight = F.softmax(tensor_atten_score, dim=-1)
        attn_weight = attn_weight.numpy()

    # attention value
    attn_value = np.zeros((SEQ, EMBD)).astype(np.float32)
    for k in range(N_HEAD):
        linear_projection(
            (
                attn_weight[:, k * SEQ : (k + 1) * SEQ]
                if USE_ALL_NPU_KERNELS
                else attn_weight[:, k, :]
            ),
            value[:, k * HEAD_DIM : (k + 1) * HEAD_DIM],
            attn_value[:, k * HEAD_DIM : (k + 1) * HEAD_DIM],
            SEQ,
            SEQ,
            HEAD_DIM,
        )
    # output projection
    x = np.zeros((SEQ, EMBD)).astype(np.float32)
    linear_projection(attn_value, params["Wo"], x, SEQ, EMBD, EMBD)
    # add residual
    add_residual(residual, x, SEQ, EMBD)
    # norm
    layernorm(residual, params["W_norm_2"], params["b_norm_2"], x)
    # up projection
    ffn_up_x = np.zeros((SEQ, FFN_HID)).astype(np.float32)
    linear_projection(x, params["W_up"], ffn_up_x, SEQ, FFN_HID, EMBD)

    if USE_ALL_NPU_KERNELS:
        activeated_x = np.zeros((SEQ, FFN_HID)).astype(np.float32)
        for i in range(SEQ // GELU_SEQ_TILE):
            gelu_mod(
                ffn_up_x[i * GELU_SEQ_TILE : (i + 1) * GELU_SEQ_TILE, :],
                activeated_x[i * GELU_SEQ_TILE : (i + 1) * GELU_SEQ_TILE, :],
            )
    else:
        tensor_ffn_up_x = torch.from_numpy(ffn_up_x)
        gelu_func = nn.GELU()
        activeated_x = gelu_func(tensor_ffn_up_x).numpy()

    x = np.zeros((SEQ, EMBD)).astype(np.float32)
    linear_projection(activeated_x, params["W_down"], x, SEQ, EMBD, FFN_HID)
    add_residual(residual, x, SEQ, EMBD)
    return residual


if __name__ == "__main__":
    ref_model = MiniGPT2().eval()
    # reference weights (float32)
    p = {n: v.detach().numpy() for n, v in ref_model.named_parameters()}
    params_fp32 = {
        "Wq": p["attn.in_proj_weight"][:EMBD, :].T,
        "Wk": p["attn.in_proj_weight"][EMBD : 2 * EMBD, :].T,
        "Wv": p["attn.in_proj_weight"][2 * EMBD :, :].T,
        "Wo": p["attn.out_proj.weight"].T,
        "W_up": p["ffn_up.weight"].T,
        "W_down": p["ffn_down.weight"].T,
        "W_norm_1": p["ln_1.weight"],
        "b_norm_1": p["ln_1.bias"],
        "W_norm_2": p["ln_2.weight"],
        "b_norm_2": p["ln_2.bias"],
    }

    params = {
        k: v.astype(np.float32) if isinstance(v, np.ndarray) else v
        for k, v in params_fp32.items()
    }

    # random input
    x_float = torch.randn(SEQ, EMBD)
    # test
    sample = ref_model(x_float)
    allo_out = run(x_float.numpy(), params)
    np.testing.assert_allclose(allo_out, sample.detach().numpy(), rtol=1e-1)
    print("Allo float32 block matches PyTorch float32 reference within tolerance ✔️")
