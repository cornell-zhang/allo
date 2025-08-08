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
    operand_qk = ExternalModule(
        top="operand_qk",
        impl_path=KERNEL_LIB_PATH + "fa_components.cc",
        input_idx=[0, 1],
        output_idx=[2],
    )

    Ty = float32
    Ly = Layout("S0")

    @df.region()
    def top():
        k_pipe = df.array(
            df.pipe(dtype=Ty, shape=(chunk_size, HEAD_DIM), depth=2),
            shape=(SEQ_LEN // chunk_size,),
        )
        v_pipe = df.array(
            df.pipe(dtype=Ty, shape=(chunk_size, HEAD_DIM), depth=2),
            shape=(SEQ_LEN // chunk_size,),
        )
        o_pipe = df.array(
            df.pipe(dtype=Ty, shape=(chunk_size, HEAD_DIM), depth=2),
            shape=(SEQ_LEN // chunk_size,),
        )

        @df.kernel(mapping=[1])
        def flash_attn_block(Q: Ty[chunk_size, HEAD_DIM]):
            max_logit: Ty[chunk_size] = 0
            with allo.meta_for(SEQ_LEN // chunk_size) as i:
                # TODO: attn score
                logits: Ty[chunk_size, chunk_size] = 0
                operand_qk(Q, k_pipe.get(), max_logit, logits, max_logit)

        @df.kernel(mapping=[SEQ_LEN // chunk_size])
        def send_k(K: Ty[SEQ_LEN, HEAD_DIM] @ Ly):
            k_pipe.put(K)

        @df.kernel(mapping=[SEQ_LEN // chunk_size])
        def send_v(V: Ty[SEQ_LEN, HEAD_DIM] @ Ly):
            v_pipe.put(V)

        @df.kernel(mapping=[1])
        def flash_attn_acc(O: Ty[chunk_size, HEAD_DIM]):
            sum_exp: Ty[chunk_size] = 0
            attn_output: Ty[chunk_size, HEAD_DIM] = 0
            # TODO

            O[:] = attn_output


N, D = 128, 64  # Sequence Length, Embedding Dim = 64
Q = np.random.randn(N, D).astype(np.float32)
K = np.random.randn(N, D).astype(np.float32)
V = np.random.randn(N, D).astype(np.float32)

out = flash_attention(Q, K, V, chunk_size=32)
print(out.shape)
print(out)
