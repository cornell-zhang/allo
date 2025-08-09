import os
import allo
from allo.ir.types import float32
import allo.dataflow as df
import numpy as np
from allo.memory import Layout
from allo.backend.experimental.external_kernel import ExternalModule

KERNEL_LIB_PATH = "./exp_kernels/"


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
    N, D = Q.shape
    output = np.zeros((N, D), dtype=Q.dtype)

    for q_start in range(0, N, chunk_size):
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
            max_logit = np.maximum(max_logit, local_max)
            exp_logits = np.exp(logits - max_logit)

            V_chunk = V[k_start:k_end, :]  # (ck, D)
            sum_exp += exp_logits.sum(axis=1, keepdims=True)
            acc += exp_logits @ V_chunk

        output[q_start:q_end, :] = acc / sum_exp

    return output


def test_flash_attention(SEQ_LEN, HEAD_DIM, chunk_size):
    attn_score = ExternalModule(
        top="transpose_matmul_with_scale",
        impl_path=KERNEL_LIB_PATH + "fa_components.cc",
        input_idx=[0, 1],
        output_idx=[2],
    )

    init_softmax = ExternalModule(
        top="init_softmax",
        impl_path=KERNEL_LIB_PATH + "fa_components.cc",
        input_idx=[],
        output_idx=[0, 1],
    )

    online_softmax = ExternalModule(
        top="online_softmax",
        impl_path=KERNEL_LIB_PATH + "fa_components.cc",
        input_idx=[0, 1, 2],
        output_idx=[3, 4, 5],
    )

    scale_attn_output = ExternalModule(
        top="scale_attn_output",
        impl_path=KERNEL_LIB_PATH + "fa_components.cc",
        input_idx=[0, 1],
        output_idx=[2],
    )

    Ty = float32
    Ly = Layout("S0R")

    @df.region()
    def top():
        q_pipe = df.array(
            df.pipe(dtype=Ty, shape=(chunk_size, HEAD_DIM), depth=2),
            shape=(SEQ_LEN // chunk_size,),
        )

        score_pipe = df.array(
            df.pipe(dtype=Ty, shape=(chunk_size, chunk_size), depth=2),
            shape=(SEQ_LEN // chunk_size,),
        )

        weight_pipe = df.array(
            df.pipe(dtype=Ty, shape=(chunk_size, chunk_size), depth=2),
            shape=(SEQ_LEN // chunk_size,),
        )

        o_pipe = df.array(
            df.pipe(dtype=Ty, shape=(chunk_size, HEAD_DIM), depth=2),
            shape=(SEQ_LEN // chunk_size,),
        )

        exp_sum_pipe = df.pipe(dtype=Ty, shape=(chunk_size,), depth=2)

        @df.kernel(mapping=[1])
        def send_q(Q: Ty[chunk_size, HEAD_DIM]):
            with allo.meta_for(SEQ_LEN // chunk_size) as i:
                q_pipe[i].put(Q)

        @df.kernel(mapping=[SEQ_LEN // chunk_size])
        def cal_attn_score(K: Ty[SEQ_LEN, HEAD_DIM] @ Ly):
            score: Ty[chunk_size, chunk_size]
            pi = df.get_pid()
            attn_score(q_pipe[pi].get(), K, score)
            score_pipe[pi].put(score)

        @df.kernel(mapping=[1])
        def cal_softmax():
            max_logit: Ty[chunk_size]
            sum_exp: Ty[chunk_size]
            init_softmax(max_logit, sum_exp)
            # softmax
            with allo.meta_for(SEQ_LEN // chunk_size) as i:
                attn_weight: Ty[chunk_size, chunk_size]
                online_softmax(
                    score_pipe[i].get(),
                    max_logit,
                    sum_exp,
                    attn_weight,
                    max_logit,
                    sum_exp,
                )
                weight_pipe[i].put(attn_weight)
            exp_sum_pipe.put(sum_exp)

        @df.kernel(mapping=[SEQ_LEN // chunk_size])
        def attn(V: Ty[SEQ_LEN, HEAD_DIM] @ Ly):
            pi = df.get_pid()
            o_pipe[pi].put(allo.matmul(weight_pipe[pi].get(), V))

        @df.kernel(mapping=[1])
        def acc(O: Ty[chunk_size, HEAD_DIM]):
            attn_output: Ty[chunk_size, HEAD_DIM] = 0
            with allo.meta_for(SEQ_LEN // chunk_size) as i:
                attn_output[:, :] = allo.add(attn_output, o_pipe[i].get())
            scale_attn_output(attn_output, exp_sum_pipe.get(), O)

    mod = df.build(
        top,
        target="aie-mlir",
        # mapping_primitives=[
        #     ("bundle", ["cal_attn_score_0", "cal_attn_score_1"]),
        #     ("bundle", ["attn_0", "attn_1"])
        # ],
        profile=True,
        warmup=20,
        num_iters=100,  # ! executing only once may get undefined result.
    )
    chunk_size = 32
    Q = np.random.randn(chunk_size, D).astype(np.float32)
    K = np.random.randn(N, D).astype(np.float32)
    V = np.random.randn(N, D).astype(np.float32)
    O = np.zeros(chunk_size * D).astype(np.float32)
    mod(Q, K, V, O)


N, D = 64, 64  # Sequence Length, Embedding Dim = 64
chunk_size = 32
# Q = np.random.randn(N, D).astype(np.float32)
# K = np.random.randn(N, D).astype(np.float32)
# V = np.random.randn(N, D).astype(np.float32)

# out = flash_attention(Q, K, V, chunk_size=32)
# print(out.shape)
# print(out)

test_flash_attention(N, D, chunk_size)
