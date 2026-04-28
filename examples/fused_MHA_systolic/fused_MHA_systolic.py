# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
import numpy as np
from allo.ir.types import float32, Int, Stream
import allo.dataflow as df
import allo.backend.hls as hls

int8 = Int(8)
int32 = Int(32)


def get_systolic_top(
    BATCH_SIZE: int,
    CONTEXT_LENGTH: int,
    HIDDEN_SIZE: int,
    NUM_HEADS: int,
    BLOCK_T: int,
):
    HEAD_DIM = HIDDEN_SIZE // NUM_HEADS
    NUM_TC = CONTEXT_LENGTH // BLOCK_T

    assert NUM_TC == BLOCK_T, "This design requires NUM_TC == BLOCK_T"

    P0 = BLOCK_T + 2
    P1 = BLOCK_T + 2

    D_SQRT = float(HEAD_DIM**0.5)
    D = 1.0 / D_SQRT
    THREE_H = 3 * HIDDEN_SIZE
    IN_ELEMS = BATCH_SIZE * CONTEXT_LENGTH * THREE_H
    OUT_ELEMS = BATCH_SIZE * CONTEXT_LENGTH * NUM_HEADS * HEAD_DIM

    @df.region()
    def top(
        input_mem: float32[IN_ELEMS],
        output_mem: float32[OUT_ELEMS],
    ):
        fifo_Q: Stream[int8[HEAD_DIM], 256][P0, P1]
        fifo_K: Stream[int8[HEAD_DIM], 256][P0, P1]
        fifo_V: Stream[float32[HEAD_DIM], 256][P0, P1]
        fifo_dQ: Stream[float32, 32][P0, P1]
        fifo_dK: Stream[float32, 32][P0, P1]
        fifo_m: Stream[float32, 32][P0, P1]
        fifo_d: Stream[float32, 32][P0, P1]
        fifo_o: Stream[float32[HEAD_DIM], 256][P0, P1]

        fifo_in_Q: Stream[int8[HEAD_DIM], 256][P0 - 2]
        fifo_in_dQ: Stream[float32, 32][P0 - 2]
        fifo_in_K: Stream[int8[HEAD_DIM], 256][P1 - 2]
        fifo_in_dK: Stream[float32, 32][P1 - 2]
        fifo_in_V: Stream[float32[HEAD_DIM], 256][P1 - 2]
        fifo_out: Stream[float32[HEAD_DIM], 256][P0 - 2]

        @df.kernel(mapping=[1], args=[input_mem])
        def load(input_d: float32[IN_ELEMS]):

            for b, h in allo.grid(BATCH_SIZE, NUM_HEADS):

                # ── Pre-pass: compute mean(K) ──────────────────────────────
                k_mean: float32[HEAD_DIM] = 0
                for t in range(CONTEXT_LENGTH):
                    for jj in range(HEAD_DIM):
                        k_idx = (
                            b * (CONTEXT_LENGTH * THREE_H)
                            + t * THREE_H
                            + 1 * HIDDEN_SIZE
                            + h * HEAD_DIM
                            + jj
                        )
                        k_mean[jj] += input_d[k_idx]
                for jj in range(HEAD_DIM):
                    # annotation drives implicit float16/int division
                    inv_ctx: float32 = 1.0 / CONTEXT_LENGTH
                    k_mean[jj] = k_mean[jj] * inv_ctx

                for tr in range(0, CONTEXT_LENGTH, BLOCK_T):

                    # ── Q: fold 1/√d, per-vector INT8 quantization ────────
                    for i in range(BLOCK_T):
                        q_scaled: float32[HEAD_DIM] = 0
                        q_abs_max: float32 = 1e-8
                        t_q = tr + i
                        for jj in range(HEAD_DIM):
                            q_idx = (
                                b * (CONTEXT_LENGTH * THREE_H)
                                + t_q * THREE_H
                                + 0 * HIDDEN_SIZE
                                + h * HEAD_DIM
                                + jj
                            )
                            d_f16: float32 = D  # typed literal
                            val: float32 = input_d[q_idx] * d_f16
                            q_scaled[jj] = val
                            neg_val: float32 = -val
                            abs_val: float32 = val if val >= 0.0 else neg_val
                            if abs_val > q_abs_max:
                                q_abs_max = abs_val

                        inv127: float32 = 1.0 / 127.0
                        delta_Q: float32 = q_abs_max * inv127

                        # ── implicit float16→int8 via typed target ─────────
                        q_int8: int8[HEAD_DIM] = 0
                        for jj in range(HEAD_DIM):
                            # Allo sees target is int8, inserts cast from float16
                            q_int8[jj] = q_scaled[jj] / delta_Q

                        if i == 0:
                            fifo_in_Q[0].put(q_int8)
                            fifo_in_dQ[0].put(delta_Q)
                        elif i == 1:
                            fifo_in_Q[1].put(q_int8)
                            fifo_in_dQ[1].put(delta_Q)
                        elif i == 2:
                            fifo_in_Q[2].put(q_int8)
                            fifo_in_dQ[2].put(delta_Q)
                        elif i == 3:
                            fifo_in_Q[3].put(q_int8)
                            fifo_in_dQ[3].put(delta_Q)

                    # ── K: smooth + per-block INT8 quantization ───────────
                    for tc_b in range(NUM_TC):

                        k_block: float32[BLOCK_T, HEAD_DIM] = 0
                        k_abs_max: float32 = 1e-8

                        for i in range(BLOCK_T):
                            t_k = tc_b * BLOCK_T + i
                            for jj in range(HEAD_DIM):
                                k_idx = (
                                    b * (CONTEXT_LENGTH * THREE_H)
                                    + t_k * THREE_H
                                    + 1 * HIDDEN_SIZE
                                    + h * HEAD_DIM
                                    + jj
                                )
                                val: float32 = input_d[k_idx] - k_mean[jj]
                                k_block[i, jj] = val
                                neg_val: float32 = -val
                                abs_val: float32 = val if val >= 0.0 else neg_val
                                if abs_val > k_abs_max:
                                    k_abs_max = abs_val

                        inv127k: float32 = 1.0 / 127.0
                        delta_K: float32 = k_abs_max * inv127k

                        if tc_b == 0:
                            fifo_in_dK[0].put(delta_K)
                        elif tc_b == 1:
                            fifo_in_dK[1].put(delta_K)
                        elif tc_b == 2:
                            fifo_in_dK[2].put(delta_K)
                        elif tc_b == 3:
                            fifo_in_dK[3].put(delta_K)

                        for i in range(BLOCK_T):
                            # ── implicit float16→int8 via typed target ─────
                            k_int8: int8[HEAD_DIM] = 0
                            for jj in range(HEAD_DIM):
                                k_int8[jj] = k_block[i, jj] / delta_K

                            v_vec: float32[HEAD_DIM] = 0
                            t_k = tc_b * BLOCK_T + i
                            for jj in range(HEAD_DIM):
                                v_idx = (
                                    b * (CONTEXT_LENGTH * THREE_H)
                                    + t_k * THREE_H
                                    + 2 * HIDDEN_SIZE
                                    + h * HEAD_DIM
                                    + jj
                                )
                                v_vec[jj] = input_d[v_idx]

                            if tc_b == 0:
                                fifo_in_K[0].put(k_int8)
                                fifo_in_V[0].put(v_vec)
                            elif tc_b == 1:
                                fifo_in_K[1].put(k_int8)
                                fifo_in_V[1].put(v_vec)
                            elif tc_b == 2:
                                fifo_in_K[2].put(k_int8)
                                fifo_in_V[2].put(v_vec)
                            elif tc_b == 3:
                                fifo_in_K[3].put(k_int8)
                                fifo_in_V[3].put(v_vec)

        @df.kernel(mapping=[P0, P1], args=[])
        def pe():
            i, j = df.get_pid()

            with allo.meta_if(i in {0, P0 - 1} and j in {0, P1 - 1}):
                pass

            with allo.meta_elif(i == 0):
                for b, h in allo.grid(BATCH_SIZE, NUM_HEADS):
                    for tr in range(0, CONTEXT_LENGTH, BLOCK_T):
                        dK: float32 = fifo_in_dK[j - 1].get()
                        fifo_dK[i + 1, j].put(dK)
                        for k in range(BLOCK_T):
                            kv: int8[HEAD_DIM] = fifo_in_K[j - 1].get()
                            vv: float32[HEAD_DIM] = fifo_in_V[j - 1].get()
                            fifo_K[i + 1, j].put(kv)
                            fifo_V[i + 1, j].put(vv)

            with allo.meta_elif(j == 0):
                for b, h in allo.grid(BATCH_SIZE, NUM_HEADS):
                    for tr in range(0, CONTEXT_LENGTH, BLOCK_T):
                        q_int8: int8[HEAD_DIM] = fifo_in_Q[i - 1].get()
                        dQ: float32 = fifo_in_dQ[i - 1].get()
                        fifo_Q[i, j + 1].put(q_int8)
                        fifo_dQ[i, j + 1].put(dQ)
                        m_init: float32 = -1e4  # float16 safe large neg
                        d_init: float32 = 0.0
                        o_init: float32[HEAD_DIM] = 0
                        fifo_m[i, j + 1].put(m_init)
                        fifo_d[i, j + 1].put(d_init)
                        fifo_o[i, j + 1].put(o_init)

            with allo.meta_elif(i == P0 - 1):
                for b, h in allo.grid(BATCH_SIZE, NUM_HEADS):
                    for tr in range(0, CONTEXT_LENGTH, BLOCK_T):
                        _dk: float32 = fifo_dK[i, j].get()
                        for k in range(BLOCK_T):
                            _k: int8[HEAD_DIM] = fifo_K[i, j].get()
                            _v: float32[HEAD_DIM] = fifo_V[i, j].get()

            with allo.meta_elif(j == P1 - 1):
                for b, h in allo.grid(BATCH_SIZE, NUM_HEADS):
                    for tr in range(0, CONTEXT_LENGTH, BLOCK_T):
                        _q: int8[HEAD_DIM] = fifo_Q[i, j].get()
                        _dq: float32 = fifo_dQ[i, j].get()
                        _m: float32 = fifo_m[i, j].get()
                        _d: float32 = fifo_d[i, j].get()
                        o_final: float32[HEAD_DIM] = fifo_o[i, j].get()
                        fifo_out[i - 1].put(o_final)

            with allo.meta_else():
                for b, h in allo.grid(BATCH_SIZE, NUM_HEADS):
                    for tr in range(0, CONTEXT_LENGTH, BLOCK_T):

                        q_int8: int8[HEAD_DIM] = fifo_Q[i, j].get()
                        dQ: float32 = fifo_dQ[i, j].get()
                        m_cur: float32 = fifo_m[i, j].get()
                        d_cur: float32 = fifo_d[i, j].get()
                        o_cur: float32[HEAD_DIM] = fifo_o[i, j].get()

                        dK: float32 = fifo_dK[i, j].get()
                        fifo_dK[i + 1, j].put(dK)

                        for k in range(BLOCK_T):
                            k_int8: int8[HEAD_DIM] = fifo_K[i, j].get()
                            vv: float32[HEAD_DIM] = fifo_V[i, j].get()

                            # ── INT8 × INT8 → INT32 accumulator ───────────
                            # Declare int32 intermediates — Allo auto-widens int8
                            x_int32: int32 = 0
                            for dd in range(HEAD_DIM):
                                qi: int32 = q_int8[dd]  # int8 → int32 via annotation
                                ki: int32 = k_int8[dd]  # int8 → int32 via annotation
                                x_int32 += qi * ki

                            # ── Dequantize INT32 → float32 via annotation ──
                            x_f32: float32 = x_int32  # int32 → float32 via annotation
                            x: float32 = x_f32 * dQ * dK

                            # ── Online softmax — all float32 ───────────────
                            m_new: float32 = m_cur
                            if x > m_cur:
                                m_new = x
                            ep: float32 = allo.exp(m_cur - m_new)
                            ex: float32 = allo.exp(x - m_new)
                            d_new: float32 = d_cur * ep + ex
                            al: float32 = d_cur * ep / d_new
                            be: float32 = ex / d_new

                            # ── P·V: float32 × float32 ─────────────────────
                            o_new: float32[HEAD_DIM] = 0
                            for dd in range(HEAD_DIM):
                                o_new[dd] = o_cur[dd] * al + be * vv[dd]

                            fifo_K[i + 1, j].put(k_int8)
                            fifo_V[i + 1, j].put(vv)

                            m_cur = m_new
                            d_cur = d_new
                            for dd in range(HEAD_DIM):
                                o_cur[dd] = o_new[dd]

                        fifo_Q[i, j + 1].put(q_int8)
                        fifo_dQ[i, j + 1].put(dQ)
                        fifo_m[i, j + 1].put(m_cur)
                        fifo_d[i, j + 1].put(d_cur)
                        fifo_o[i, j + 1].put(o_cur)

        @df.kernel(mapping=[1], args=[output_mem])
        def store(global_mem: float32[OUT_ELEMS]):
            for b, h in allo.grid(BATCH_SIZE, NUM_HEADS):
                for tr in range(0, CONTEXT_LENGTH, BLOCK_T):
                    for i in range(BLOCK_T):
                        if i == 0:
                            o0: float32[HEAD_DIM] = fifo_out[0].get()
                            for jj in range(HEAD_DIM):
                                idx = (
                                    (b * CONTEXT_LENGTH + (tr + i)) * NUM_HEADS + h
                                ) * HEAD_DIM + jj
                                global_mem[idx] = o0[jj]
                        elif i == 1:
                            o1: float32[HEAD_DIM] = fifo_out[1].get()
                            for jj in range(HEAD_DIM):
                                idx = (
                                    (b * CONTEXT_LENGTH + (tr + i)) * NUM_HEADS + h
                                ) * HEAD_DIM + jj
                                global_mem[idx] = o1[jj]
                        elif i == 2:
                            o2: float32[HEAD_DIM] = fifo_out[2].get()
                            for jj in range(HEAD_DIM):
                                idx = (
                                    (b * CONTEXT_LENGTH + (tr + i)) * NUM_HEADS + h
                                ) * HEAD_DIM + jj
                                global_mem[idx] = o2[jj]
                        elif i == 3:
                            o3: float32[HEAD_DIM] = fifo_out[3].get()
                            for jj in range(HEAD_DIM):
                                idx = (
                                    (b * CONTEXT_LENGTH + (tr + i)) * NUM_HEADS + h
                                ) * HEAD_DIM + jj
                                global_mem[idx] = o3[jj]

    return top
