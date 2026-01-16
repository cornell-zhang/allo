# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import torch
import math
import torch.nn.functional as F
import allo.dataflow as df
import numpy as np
from ml_dtypes import bfloat16 as np_bfloat16
from allo.library.aie.modules.flash_attn import FA


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


def _test_flash_attention(
    SEQ_LEN, HEAD_DIM, Q_tile_size, q_chunk_size=32, kv_chunk_size=32
):

    top, mapping_primitives_ = FA(
        SEQ_LEN, HEAD_DIM, Q_tile_size, q_chunk_size, kv_chunk_size
    )
    mod = df.build(
        top,
        project=f"fa_{SEQ_LEN}.prj",
        target="aie",
        mapping_primitives=mapping_primitives_,
        profile=True,
        warmup=20,
        num_iters=100,
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
    os.environ["ALLO_EXTERNAL_KERNEL_DIR"] = (
        f"{dir_path}/../../allo/library/aie/kernels/"
    )
    os.environ["ENABLE_AGGRESSIVE_PORT_UTILIZATION_PATCH"] = "1"
    os.environ["COALESCE_MORE"] = "1"
    os.environ["FORCE_UNROLL_INDEX"] = "0"

    seq_len_list = [64, 128, 256, 512, 1024, 2048]
    for seq_len in seq_len_list:
        _test_flash_attention(seq_len, 64, seq_len, q_chunk_size=32, kv_chunk_size=32)

    del os.environ["ALLO_EXTERNAL_KERNEL_DIR"]
    del os.environ["ENABLE_AGGRESSIVE_PORT_UTILIZATION_PATCH"]
    del os.environ["COALESCE_MORE"]
    del os.environ["FORCE_UNROLL_INDEX"]
