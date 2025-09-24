import os
import allo
from allo.ir.types import bfloat16
import allo.dataflow as df
import numpy as np
from ml_dtypes import bfloat16 as np_bfloat16
from allo.memory import Layout
from allo.backend.aie.external_kernel import ExternalModule

KERNEL_LIB_PATH = "/home/sf668/workspace/allo/tests/dataflow/aie/gpt2/kernels/"
np.random.seed(42)

# ################################################################
# Flash Attention
# ################################################################


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
    N, D = K.shape
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


def test_flash_attention(SEQ_LEN, HEAD_DIM, q_chunk_size=32, kv_chunk_size=32):
    COL = 4
    iteration = SEQ_LEN // q_chunk_size

    init_softmax = ExternalModule(
        top="init_softmax",
        impl_path=KERNEL_LIB_PATH + "lut_softmax.cc",
        input_idx=[],
        output_idx=[0, 1],
    )

    online_softmax = ExternalModule(
        top="online_softmax",
        impl_path=KERNEL_LIB_PATH + "lut_softmax.cc",
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
        q_pipe = df.array(
            df.pipe(dtype=Ty, shape=(q_chunk_size, HEAD_DIM), depth=2),
            shape=(SEQ_LEN // q_chunk_size, SEQ_LEN // kv_chunk_size),
        )

        score_pipe = df.array(
            df.pipe(dtype=Ty, shape=(q_chunk_size, kv_chunk_size), depth=2),
            shape=(SEQ_LEN // q_chunk_size, SEQ_LEN // kv_chunk_size),
        )

        weight_pipe = df.array(
            df.pipe(dtype=Ty, shape=(q_chunk_size, kv_chunk_size), depth=2),
            shape=(SEQ_LEN // q_chunk_size, SEQ_LEN // kv_chunk_size),
        )

        o_pipe = df.array(
            df.pipe(dtype=Ty, shape=(q_chunk_size, HEAD_DIM), depth=2),
            shape=(SEQ_LEN // q_chunk_size, SEQ_LEN // kv_chunk_size),
        )

        exp_sum_pipe = df.array(
            df.pipe(dtype=Ty, shape=(kv_chunk_size,), depth=2),
            shape=(SEQ_LEN // q_chunk_size,),
        )

        exp_scale_pipe = df.array(
            df.pipe(dtype=Ty, shape=(kv_chunk_size,), depth=2),
            shape=(SEQ_LEN // q_chunk_size,),
        )

        @df.kernel(mapping=[SEQ_LEN // q_chunk_size, 1])
        def send_q(Q: Ty[SEQ_LEN, HEAD_DIM] @ Ly_outer):
            po, _ = df.get_pid()
            with allo.meta_for(SEQ_LEN // kv_chunk_size) as i:
                q_pipe[po, i].put(Q)

        @df.kernel(mapping=[SEQ_LEN // q_chunk_size, SEQ_LEN // kv_chunk_size])
        def cal_attn_score(K: Ty[HEAD_DIM, SEQ_LEN] @ Ly_K):
            po, pi = df.get_pid()
            score: Ty[q_chunk_size, kv_chunk_size] = allo.matmul(
                q_pipe[po, pi].get(), K
            )
            score_pipe[po, pi].put(score)

        @df.kernel(mapping=[SEQ_LEN // q_chunk_size, 1])
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

        @df.kernel(mapping=[SEQ_LEN // q_chunk_size, SEQ_LEN // kv_chunk_size])
        def attn(V: Ty[SEQ_LEN, HEAD_DIM] @ Ly_inner):
            po, pi = df.get_pid()
            o_pipe[po, pi].put(allo.matmul(weight_pipe[po, pi].get(), V))

        @df.kernel(mapping=[SEQ_LEN // q_chunk_size, 1])
        def acc(O: Ty[SEQ_LEN, HEAD_DIM] @ Ly_outer):
            attn_output: Ty[q_chunk_size, HEAD_DIM] = 0
            po, _ = df.get_pid()
            with allo.meta_for(SEQ_LEN // kv_chunk_size) as i:
                rescale_attn_output(attn_output, exp_scale_pipe[po].get(), attn_output)
                attn_output[:, :] = allo.add(attn_output, o_pipe[po, i].get())
            scale_attn_output(attn_output, exp_sum_pipe[po].get(), O)

    mapping_primitives_ = []
    for idx in range(iteration):
        nodes = [f"cal_attn_score_{idx}_{i}" for i in range(SEQ_LEN // kv_chunk_size)]
        mapping_primitives_.append(("bundle", nodes))
        nodes = [f"attn_{idx}_{i}" for i in range(SEQ_LEN // kv_chunk_size)]
        mapping_primitives_.append(("bundle", nodes))
        mapping_primitives_.append(
            ("chain", [f"send_q_{idx}_0", f"cal_attn_score_{idx}_0"])
        )
    if iteration // COL > 1:
        for i_ in range(COL):
            mapping_primitives_.append(
                (
                    "bundle",
                    [
                        f"send_q_{idx*COL+i_}_0-cal_attn_score_{idx*COL+i_}_0"
                        for idx in range(iteration // COL)
                    ],
                )
            )
            mapping_primitives_.append(
                (
                    "bundle",
                    [f"cal_softmax_{idx*COL+i_}_0" for idx in range(iteration // COL)],
                )
            )
            mapping_primitives_.append(
                ("bundle", [f"attn_{idx*COL+i_}_0" for idx in range(iteration // COL)])
            )
            mapping_primitives_.append(
                ("bundle", [f"acc_{idx*COL+i_}_0" for idx in range(iteration // COL)])
            )

    # print(mapping_primitives)
    mod = df.build(
        top,
        project=f"fa_{SEQ_LEN}.prj",
        target="aie-mlir",
        # mapping_primitives=mapping_primitives_,
        profile=False,
        warmup=20,
        num_iters=100,
        # device_type="npu1_2col",
    )
    Q = np.random.randn(SEQ_LEN, HEAD_DIM)
    K = np.random.randn(SEQ_LEN, HEAD_DIM)
    V = np.random.randn(SEQ_LEN, HEAD_DIM)
    Q_ = Q.astype(np_bfloat16)
    K_ = K.astype(np_bfloat16)
    V_ = V.astype(np_bfloat16)
    O = np.zeros(SEQ_LEN * HEAD_DIM).astype(np_bfloat16)
    mod(Q_, K_.T, V_, O)
    np.set_printoptions(threshold=np.inf)
    out = flash_attention(Q_, K_, V_, chunk_size=32)
    print(out)
    O = O.astype(np.float32).reshape(SEQ_LEN, HEAD_DIM)
    print(O)


if __name__ == "__main__":
    # N, D = 1024, 64  # Sequence Length, Embedding Dim = 64
    # chunk_size = 32
    # print(out.shape)
    # print(out)
    # seq_len_list = [128,256,512,1024,2048]
    seq_len_list = [32]
    for seq_len in seq_len_list:
        test_flash_attention(seq_len, 64, q_chunk_size=32, kv_chunk_size=32)
