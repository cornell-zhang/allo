# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import allo
from allo.ir.types import bfloat16, Stream
import allo.dataflow as df
from allo.memory import Layout
from allo.utils import construct_kernel_name
from allo.backend.aie.external_kernel import ExternalModule

S = Layout.Shard
R = Layout.Replicate


def FA(SEQ_LEN, HEAD_DIM, Q_tile_size, q_chunk_size, kv_chunk_size):
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
    Ly_outer = [S(0), R]
    Ly_inner = [S(1), R]
    Ly_K = [R, S(1)]

    @df.region()
    def top(
        Q: Ty[Q_tile_size, HEAD_DIM],
        K: Ty[HEAD_DIM, SEQ_LEN],
        V: Ty[SEQ_LEN, HEAD_DIM],
        O: Ty[Q_tile_size, HEAD_DIM],
    ):
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

        @df.kernel(mapping=[Q_tile_size // q_chunk_size, 1], args=[Q])
        def send_q(local_Q: Ty[Q_tile_size, HEAD_DIM] @ Ly_outer):
            po, _ = df.get_pid()
            with allo.meta_for(SEQ_LEN // kv_chunk_size) as i:
                q_pipe[po, i].put(local_Q)

        @df.kernel(
            mapping=[Q_tile_size // q_chunk_size, SEQ_LEN // kv_chunk_size], args=[K]
        )
        def cal_attn_score(local_K: Ty[HEAD_DIM, SEQ_LEN] @ Ly_K):
            po, pi = df.get_pid()
            score: Ty[q_chunk_size, kv_chunk_size] = allo.matmul(
                q_pipe[po, pi].get(), local_K
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

        @df.kernel(
            mapping=[Q_tile_size // q_chunk_size, SEQ_LEN // kv_chunk_size], args=[V]
        )
        def attn(local_V: Ty[SEQ_LEN, HEAD_DIM] @ Ly_inner):
            po, pi = df.get_pid()
            o_pipe[po, pi].put(allo.matmul(weight_pipe[po, pi].get(), local_V))

        @df.kernel(mapping=[Q_tile_size // q_chunk_size, 1], args=[O])
        def acc(local_O: Ty[Q_tile_size, HEAD_DIM] @ Ly_outer):
            attn_output: Ty[q_chunk_size, HEAD_DIM] = 0
            po, _ = df.get_pid()
            with allo.meta_for(SEQ_LEN // kv_chunk_size) as i:
                rescale_attn_output(attn_output, exp_scale_pipe[po].get(), attn_output)
                attn_output[:, :] = allo.add(attn_output, o_pipe[po, i].get())
            scale_attn_output(attn_output, exp_sum_pipe[po].get(), local_O)

    mapping_primitives_ = []
    sub_graphs = []
    chunks = SEQ_LEN // kv_chunk_size
    for idx in range(iteration):
        sender = construct_kernel_name("send_q", (idx, 0))
        score = construct_kernel_name("cal_attn_score", (idx, 0))
        if chunks > 1:
            bundled = {}
            for prefix in ("cal_attn_score", "attn"):
                nodes = [construct_kernel_name(prefix, (idx, i)) for i in range(chunks)]
                mapping_primitives_.append(("bundle", nodes))
                bundled[prefix] = f"{nodes[0]}x{chunks}"
            score = bundled["cal_attn_score"]
            sub_graph = (
                "-".join((sender, score)),
                construct_kernel_name("cal_softmax", (idx, 0)),
                bundled["attn"],
                construct_kernel_name("acc", (idx, 0)),
            )
        else:
            sub_graph = ("-".join((sender, score)),)
        mapping_primitives_.append(("chain", [sender, score]))
        sub_graphs.append(sub_graph)
    col_num = 4
    if iteration > col_num:
        for idx in range(col_num):
            nodes = [sub_graphs[idx + i * col_num] for i in range(iteration // col_num)]
            mapping_primitives_.append(("bundle", nodes))

    return top, mapping_primitives_
