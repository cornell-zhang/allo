# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import float32, int32, bool, index


def get_scheduled_flash_attention(
    BATCH_SIZE: int,
    CONTEXT_LENGTH: int,
    HIDDEN_SIZE: int,
    NUM_HEADS: int,
    BLOCK_T: int = 4,
):
    HEAD_DIM = HIDDEN_SIZE // NUM_HEADS
    D_SQRT = float(HEAD_DIM**0.5)
    THREE_H = 3 * HIDDEN_SIZE
    IN_ELEMS = BATCH_SIZE * CONTEXT_LENGTH * THREE_H
    OUT_ELEMS = BATCH_SIZE * CONTEXT_LENGTH * NUM_HEADS * HEAD_DIM

    def load_tile(
        global_mem: float32[IN_ELEMS],
        local_tile: float32[BLOCK_T, HEAD_DIM],
        b: index,
        h: index,
        t_start: index,
        type_offset: int32,
    ):
        for t, d in allo.grid(BLOCK_T, HEAD_DIM):
            t_global: index = t_start + t
            global_c: index = type_offset * HIDDEN_SIZE + h * HEAD_DIM + d
            idx: index = b * (CONTEXT_LENGTH * THREE_H) + t_global * THREE_H + global_c
            local_tile[t, d] = global_mem[idx]

    def store_tile(
        global_mem: float32[OUT_ELEMS],
        local_tile: float32[BLOCK_T, HEAD_DIM],
        b: index,
        h: index,
        t_start: index,
    ):
        for t, d in allo.grid(BLOCK_T, HEAD_DIM):
            t_global: index = t_start + t
            idx: index = (
                (b * CONTEXT_LENGTH + t_global) * NUM_HEADS + h
            ) * HEAD_DIM + d
            global_mem[idx] = local_tile[t, d]

    def compute_block_attention(
        Q_tile: float32[BLOCK_T, HEAD_DIM],
        K_tile: float32[BLOCK_T, HEAD_DIM],
        V_tile: float32[BLOCK_T, HEAD_DIM],
        O_tile: float32[BLOCK_T, HEAD_DIM],
        m_vec: float32[BLOCK_T],
        l_vec: float32[BLOCK_T],
        scale: float32,
        is_first_block: bool,
    ):
        S_tile: float32[BLOCK_T, BLOCK_T] = 0.0

        for i, j in allo.grid(BLOCK_T, BLOCK_T):
            for d in range(HEAD_DIM):
                S_tile[i, j] += Q_tile[i, d] * K_tile[j, d]

        for i1, j1 in allo.grid(BLOCK_T, BLOCK_T):
            S_tile[i1, j1] = S_tile[i1, j1] * scale

        for m in range(BLOCK_T):
            row_max_val: float32 = -1e30

            for n in range(BLOCK_T):
                if S_tile[m, n] > row_max_val:
                    row_max_val = S_tile[m, n]
                else:
                    row_max_val = row_max_val

            m_prev: float32 = m_vec[m]
            m_new: float32 = 0.0
            if row_max_val > m_prev:
                m_new = row_max_val
            else:
                m_new = m_prev

            alpha: float32 = 0.0
            if is_first_block:
                alpha = 0.0
            else:
                alpha = allo.exp(m_prev - m_new)
            beta: float32 = allo.exp(row_max_val - m_new)
            row_sum_exp: float32 = 0.0

            for k in range(BLOCK_T):
                row_sum_exp = row_sum_exp + allo.exp(S_tile[m, k] - row_max_val)

            l_new: float32 = (l_vec[m] * alpha) + (row_sum_exp * beta)

            p_val: float32[BLOCK_T] = 0.0
            for z in range(BLOCK_T):
                p_val[z] = allo.exp(S_tile[m, z] - row_max_val)

            for g in range(HEAD_DIM):
                pv_sum: float32 = 0.0
                for l in range(BLOCK_T):
                    pv_sum = pv_sum + p_val[l] * V_tile[l, g]
                O_tile[m, g] = O_tile[m, g] * alpha + pv_sum * beta

            m_vec[m] = m_new
            l_vec[m] = l_new

    def compute_engine(input_mem: float32[IN_ELEMS], output_mem: float32[OUT_ELEMS]):
        scale: float32 = 1.0 / D_SQRT
        Q_sram: float32[BLOCK_T, HEAD_DIM]
        K_sram: float32[BLOCK_T, HEAD_DIM]
        V_sram: float32[BLOCK_T, HEAD_DIM]
        O_sram: float32[BLOCK_T, HEAD_DIM]

        m_sram: float32[BLOCK_T]
        l_sram: float32[BLOCK_T]

        for b, h in allo.grid(BATCH_SIZE, NUM_HEADS):
            for tr in range(0, CONTEXT_LENGTH, BLOCK_T):
                load_tile(input_mem, Q_sram, b, h, tr, 0)
                for m in range(BLOCK_T):
                    m_sram[m] = -1e30
                    l_sram[m] = 0.0
                    for n in range(HEAD_DIM):
                        O_sram[m, n] = 0.0

                for tc in range(0, CONTEXT_LENGTH, BLOCK_T):
                    load_tile(input_mem, K_sram, b, h, tc, 1)
                    load_tile(input_mem, V_sram, b, h, tc, 2)
                    compute_block_attention(
                        Q_sram, K_sram, V_sram, O_sram, m_sram, l_sram, scale, (tc == 0)
                    )

                for i in range(BLOCK_T):
                    inv_l: float32 = 1.0 / (l_sram[i] + 1e-9)
                    for d in range(HEAD_DIM):
                        O_sram[i, d] = O_sram[i, d] * inv_l

                store_tile(output_mem, O_sram, b, h, tr)

    s1 = allo.customize(load_tile)
    s1.pipeline("d")
    s1.pipeline("t")

    s2 = allo.customize(store_tile)
    s2.pipeline("d")
    s2.pipeline("t")

    s3 = allo.customize(compute_block_attention)
    s3.partition(s3.Q_tile, partition_type=0, dim=0)
    s3.partition(s3.K_tile, partition_type=0, dim=0)
    s3.unroll("d")
    s3.unroll("j", factor=4)
    s3.unroll("i", factor=4)

    s3.partition(s3.S_tile, partition_type=0, dim=0)
    s3.partition(s3.O_tile, partition_type=0, dim=0)
    s3.partition(s3.V_tile, partition_type=0, dim=0)
    s3.partition(s3.p_val, partition_type=0, dim=0)
    s3.unroll("z")
    s3.unroll("j1")
    s3.unroll("i1")
    s3.pipeline("n")
    s3.unroll("k")
    s3.unroll("l")
    s3.unroll("h")
    s3.unroll("m")

    s = allo.customize(compute_engine)
    s.partition(s.O_sram, partition_type=0, dim=0)
    s.partition(s.m_sram, partition_type=0, dim=0)
    s.partition(s.l_sram, partition_type=0, dim=0)
    s.unroll("n")
    s.unroll("m")
    s.pipeline("d")

    s.compose([s1, s2, s3])

    return s
