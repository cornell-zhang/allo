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
import numpy as np
import allo
import allo.dataflow as df
from allo.ir.types import float32, bfloat16, int32
from allo.memory import Layout
from allo.backend.experimental import ExternalModule
from ml_dtypes import bfloat16 as np_bfloat16

torch.manual_seed(0)
np.random.seed(0)

# ===============================================================================
# Model Configuration
# fixme: align with https://huggingface.co/openai-community/gpt2/blob/main/config.json
# ===============================================================================
BATCH = 1
SEQ = 8
EMBD = 768
N_HEAD = 12
HEAD_DIM = EMBD // N_HEAD
FFN_HID = EMBD * 4


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

    def forward(self, x: torch.Tensor):
        residual = x
        x = self.ln_1(x)
        # attn_out, _ = self.attn(
        #     x,
        #     x,
        #     x,
        #     need_weights=False,
        #     attn_mask=torch.triu(torch.ones(SEQ, SEQ), 1).bool(),
        # )
        # x = attn_out + residual
        # residual = x
        # x = self.ln_2(x)
        # activeated_x = self.gelu(self.ffn_up(x))
        # x = self.ffn_down(activeated_x)
        # x = residual + x
        return x


# ===============================================================================
# Allo Version
# ===============================================================================

# -------------------------------- configuration --------------------------------
Ty = float32  # All tensors use float32
P0, P1 = 2, 1
# S0 / S1 shard on mesh dims, R = replicated
LyX = Layout("S0R")  # shard rows (token dim) replicate cols
LyW = Layout("RS1")  # replicate rows shard cols
LyY = Layout("S0S1")  # shard rows & cols
LyR = Layout("R")  # replicated (scalars / vectors)

BN = BATCH * N_HEAD  # 8   flattened (batch*head)
N = BATCH * SEQ  # 16   flattened (batch*seq)
NS = BATCH * N_HEAD * SEQ  # 32  flattened for attention matrices

N_local = N // P0
SEQ_local = SEQ // P0
NS_local = NS // P0
BN_local = BN // P0


def run(x_fp32: np.ndarray, params: dict):
    norm = ExternalModule(
        top="layer_norm",
        impl_path="layer_norm.cc",
        input_idx=[0, 1],
        output_idx=[2],
    )

    # LayerNorm
    NORM_TILE = SEQ // P0
    norm_io_layout = Layout("S0R")
    norm_arg_layout = Layout("R")

    @df.region()
    def layer_norm_kernel():
        pipe = df.array(
            df.pipe(dtype=Ty, shape=(NORM_TILE, EMBD), depth=1), shape=(P0,)
        )

        @df.kernel(mapping=[P0])
        def norm_no_bias(
            input_x: Ty[SEQ, EMBD] @ norm_io_layout,
            weight: Ty[EMBD] @ norm_arg_layout,
        ):
            pi = df.get_pid()
            tmp: Ty[NORM_TILE, EMBD] = 0
            norm(input_x, weight, tmp)
            pipe[pi].put(tmp)

        @df.kernel(mapping=[P0])
        def norm_add_bias(
            bias: Ty[EMBD] @ norm_arg_layout, output_x: Ty[SEQ, EMBD] @ norm_io_layout
        ):
            pi = df.get_pid()
            data = pipe[pi].get()
            output_x[:, :] = allo.add(data, bias)

    layer_norm_mod = df.build(layer_norm_kernel, target="aie-mlir", project="norm.prj")

    x = x_fp32.astype(np.float32)
    Xf = x.reshape(N, EMBD)
    Sf = np.empty((N, EMBD), dtype=np.float32)
    layer_norm_mod(Xf, params["W_norm_1"], params["b_norm_1"], Sf)
    return Sf
    # # QKV projection
    # @df.region()
    # def qkv_linear():
    #     @df.kernel(mapping=[P0, P1])
    #     def linear(
    #         A: Ty[N, in_dim] @ LyX,  # input flattened
    #         W: Ty[in_dim, out_dim] @ LyW,
    #         Y: Ty[N, out_dim] @ LyY,
    #     ):
    #         Y[:, :] = allo.matmul(A, W)


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
    x_float = torch.randn(BATCH, SEQ, EMBD)
    ref_out = ref_model(x_float).detach().numpy()
    # print(ref_out)
    # print(params['b_norm_2'].shape)
    allo_out = (run(x_float.numpy(), params)).reshape(BATCH, SEQ, EMBD)

    np.testing.assert_allclose(allo_out, ref_out, rtol=1e-2)
