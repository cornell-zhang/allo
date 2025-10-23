# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import torch
import math
import torch.nn.functional as F
import allo
from allo.ir.types import bfloat16, Stream
import allo.dataflow as df
import numpy as np
from ml_dtypes import bfloat16 as np_bfloat16
from allo.memory import Layout
from allo.backend.aie.external_kernel import ExternalModule


# ################################################################
# Flash Attention
# ################################################################


def scaled_dot_product_attention(q, k, v):
    _, D = k.shape
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)
    attn = F.softmax(scores, dim=-1)
    output = torch.matmul(attn, v)
    return output


def flash_attention(Q, K, V, chunk_size=32):
    """
    Single-batch FlashAttention

    Args: (N: sequence length, D: head dim)
        Q: (N, D) - Queries
        K: (N, D) - Keys
        V: (N, D) - Values
        chunk_size: int - how many queries to process at a time
    Returns:
        Output: (N, D)
    """
    Q_size, _ = Q.shape
    N, D = K.shape
    output = np.zeros((Q_size, D), dtype=Q.dtype)

    for q_start in range(0, Q_size, chunk_size):
        q_end = min(q_start + chunk_size, N)
        Q_chunk = Q[q_start:q_end, :]  # (cq, D)

        # Initialize numerically stable softmax components
        max_logit = np.full((Q_chunk.shape[0], 1), -np.inf)
        sum_exp = np.zeros((Q_chunk.shape[0], 1))
        acc = np.zeros((Q_chunk.shape[0], D))

        for k_start in range(0, N, chunk_size):
            k_end = min(k_start + chunk_size, N)

            K_chunk = K[k_start:k_end, :]  # (ck, D)
            logits = Q_chunk @ K_chunk.T / np.sqrt(D)  # (cq, ck)

            local_max = np.max(logits, axis=1, keepdims=True)
            new_max = np.maximum(max_logit, local_max)
            exp_logits = np.exp(logits - new_max)
            scale = np.exp(max_logit - new_max)
            sum_exp = sum_exp * scale + exp_logits.sum(axis=1, keepdims=True)

            V_chunk = V[k_start:k_end, :]  # (ck, D)
            O = exp_logits @ V_chunk

            acc = acc * scale + O
            max_logit = new_max
        output[q_start:q_end, :] = acc / sum_exp

    return output


def gen_bundle(prefix, idx, total):
    nodes = [f"{prefix}_{idx}_{i}" for i in range(total)]
    return ("bundle", nodes)


def test_flash_attention(
    SEQ_LEN, HEAD_DIM, Q_tile_size, q_chunk_size=32, kv_chunk_size=32
):
    KERNEL_LIB_PATH = os.getenv("ALLO_EXTERNAL_KERNEL_DIR")
    iteration = Q_tile_size // q_chunk_size

    init_softmax = ExternalModule(
        top="init_softmax",
        impl_path=KERNEL_LIB_PATH + "softmax_bf16.cc",
        input_idx=[],
        output_idx=[0, 1],
    )

    online_softmax = ExternalModule(
        top="online_softmax",
        impl_path=KERNEL_LIB_PATH + "softmax_bf16.cc",
        input_idx=[0, 1, 2],
        output_idx=[3, 4, 5, 6],
    )

    rescale_attn_output = ExternalModule(
        top="rescale_attn_output",
        impl_path=KERNEL_LIB_PATH + "attn_out.cc",
        input_idx=[0, 1],
        output_idx=[2],
    )

    scale_attn_output = ExternalModule(
        top="scale_attn_output",
        impl_path=KERNEL_LIB_PATH + "attn_out.cc",
        input_idx=[0, 1],
        output_idx=[2],
    )

    Ty = bfloat16
    Ly_outer = Layout("S1R")
    Ly_inner = Layout("S0R")
    Ly_K = Layout("RS0")

    @df.region()
    def top():
        q_pipe: Stream[Ty[q_chunk_size, HEAD_DIM], 2][
            Q_tile_size // q_chunk_size, SEQ_LEN // kv_chunk_size
        ]
        score_pipe: Stream[Ty[q_chunk_size, kv_chunk_size], 2][
            Q_tile_size // q_chunk_size, SEQ_LEN // kv_chunk_size
        ]
        weight_pipe: Stream[Ty[q_chunk_size, kv_chunk_size], 2][
            Q_tile_size // q_chunk_size, SEQ_LEN // kv_chunk_size
        ]
        o_pipe: Stream[Ty[q_chunk_size, HEAD_DIM], 2][
            Q_tile_size // q_chunk_size, SEQ_LEN // kv_chunk_size
        ]
        exp_sum_pipe: Stream[Ty[kv_chunk_size], 2][Q_tile_size // q_chunk_size]
        exp_scale_pipe: Stream[Ty[kv_chunk_size], 2][Q_tile_size // q_chunk_size]

        @df.kernel(mapping=[Q_tile_size // q_chunk_size, 1])
        def send_q(Q: Ty[Q_tile_size, HEAD_DIM] @ Ly_outer):
            po, _ = df.get_pid()
            with allo.meta_for(SEQ_LEN // kv_chunk_size) as i:
                q_pipe[po, i].put(Q)

        @df.kernel(mapping=[Q_tile_size // q_chunk_size, SEQ_LEN // kv_chunk_size])
        def cal_attn_score(K: Ty[HEAD_DIM, SEQ_LEN] @ Ly_K):
            po, pi = df.get_pid()
            score: Ty[q_chunk_size, kv_chunk_size] = allo.matmul(
                q_pipe[po, pi].get(), K
            )
            score_pipe[po, pi].put(score)

        @df.kernel(mapping=[Q_tile_size // q_chunk_size, 1])
        def cal_softmax():
            po, _ = df.get_pid()
            max_logit: Ty[kv_chunk_size]
            sum_exp: Ty[kv_chunk_size]
            init_softmax(max_logit, sum_exp)
            # softmax
            with allo.meta_for(SEQ_LEN // kv_chunk_size) as i:
                attn_weight: Ty[q_chunk_size, kv_chunk_size]
                scale_exp: Ty[kv_chunk_size]
                # FIXME: / sqrt(HEAD_DIM) somewhere (maybe the best choice is to do that in QK external kernel)
                online_softmax(
                    score_pipe[po, i].get(),
                    max_logit,
                    sum_exp,
                    attn_weight,
                    scale_exp,
                    max_logit,
                    sum_exp,
                )
                exp_scale_pipe[po].put(scale_exp)
                weight_pipe[po, i].put(attn_weight)
            exp_sum_pipe[po].put(sum_exp)

        @df.kernel(mapping=[Q_tile_size // q_chunk_size, SEQ_LEN // kv_chunk_size])
        def attn(V: Ty[SEQ_LEN, HEAD_DIM] @ Ly_inner):
            po, pi = df.get_pid()
            o_pipe[po, pi].put(allo.matmul(weight_pipe[po, pi].get(), V))

        @df.kernel(mapping=[Q_tile_size // q_chunk_size, 1])
        def acc(O: Ty[Q_tile_size, HEAD_DIM] @ Ly_outer):
            attn_output: Ty[q_chunk_size, HEAD_DIM] = 0
            po, _ = df.get_pid()
            with allo.meta_for(SEQ_LEN // kv_chunk_size) as i:
                rescale_attn_output(attn_output, exp_scale_pipe[po].get(), attn_output)
                attn_output[:, :] = allo.add(attn_output, o_pipe[po, i].get())
            scale_attn_output(attn_output, exp_sum_pipe[po].get(), O)

    mapping_primitives_ = []
    for idx in range(iteration):
        if SEQ_LEN // kv_chunk_size > 1:
            nodes = [
                f"cal_attn_score_{idx}_{i}" for i in range(SEQ_LEN // kv_chunk_size)
            ]
            mapping_primitives_.append(("bundle", nodes))
            nodes = [f"attn_{idx}_{i}" for i in range(SEQ_LEN // kv_chunk_size)]
            mapping_primitives_.append(("bundle", nodes))
            mapping_primitives_.append(
                (
                    "chain",
                    [
                        f"send_q_{idx}_0",
                        f"cal_attn_score_{idx}_0x{SEQ_LEN // kv_chunk_size}",
                    ],
                )
            )
        else:
            mapping_primitives_.append(
                ("chain", [f"send_q_{idx}_0", f"cal_attn_score_{idx}_0"])
            )

    mod = df.build(
        top,
        project=f"fa_{SEQ_LEN}.prj",
        target="aie",
        mapping_primitives=mapping_primitives_,
        profile=False,
        warmup=20,
        num_iters=100,
        # device_type="npu1_2col",
    )
    Q = np.random.randn(Q_tile_size, HEAD_DIM)
    K = np.random.randn(SEQ_LEN, HEAD_DIM)
    V = np.random.randn(SEQ_LEN, HEAD_DIM)
    Q_ = Q.astype(np_bfloat16)
    K_ = K.astype(np_bfloat16)
    V_ = V.astype(np_bfloat16)
    O = np.zeros(Q_tile_size * HEAD_DIM).astype(np_bfloat16)
    mod(Q_, K_.T, V_, O)
    O = O.astype(np.float32).reshape(Q_tile_size, HEAD_DIM)

    out = flash_attention(Q_, K_, V_, chunk_size=32)
    np.testing.assert_allclose(out, O, atol=5e-2)
    print("PASSED!")

    q = torch.from_numpy(Q).to(dtype=torch.bfloat16)
    k = torch.from_numpy(K).to(dtype=torch.bfloat16)
    v = torch.from_numpy(V).to(dtype=torch.bfloat16)
    out_ = scaled_dot_product_attention(q, k, v)
    np.testing.assert_allclose(out_.to(torch.float32).numpy(), O, atol=5e-2)
    print("PASSED!")


if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.abspath(__file__))
    os.environ["ALLO_EXTERNAL_KERNEL_DIR"] = f"{dir_path}/../../../../allo/library/aie/"
    os.environ["ENABLE_AGGRESSIVE_PORT_UTILIZATION_PATCH"] = "1"
    os.environ["FORCE_UNROLL_INDEX"] = "0"

    seq_len_list = [64, 128]
    for seq_len in seq_len_list:
        test_flash_attention(seq_len, 64, seq_len, q_chunk_size=32, kv_chunk_size=32)

    del os.environ["ALLO_EXTERNAL_KERNEL_DIR"]
    del os.environ["ENABLE_AGGRESSIVE_PORT_UTILIZATION_PATCH"]
    del os.environ["FORCE_UNROLL_INDEX"]
