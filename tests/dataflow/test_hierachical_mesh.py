# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import int32, float32, Stream, Stateful
import allo.dataflow as df
import numpy as np

# Dimensions and Capacities
TILE_M = 2  # Spatial dimensions (M = N = K = 2)
M = N = K = TILE_M
P0 = P1 = TILE_M + 2

MEM_SIZE = 256  # 256 float32 elements per Data Scratchpad. MT is partitioned: Tile 0 [0:128], Tile 1 [128:256] 
IMEM_SIZE = 8   # 8 instructions per Compute Tile instruction memory (int32[IMEM_SIZE * 4])
BW = 16         # Max stream interface fixed payload bandwidth

# Compute Tile Opcodes (4 words wide: opcode, addr1, addr2, addr3)
OP_NOP = 0   # No-op; skipped during execution (costs 1 compare, 0 streams).
OP_GEMM = 1  # Matrix Multiply: C = A @ B. Reads A (2x2) at addr1, B (2x2) at addr2, writes C (2x2) at addr3.
OP_VADD = 2  # Vector Addition: C = A + B. Reads A (1x2) at addr1, B (1x2) at addr2, writes C (1x2) at addr3.

# Top-Level Control Actions (ctrl[0] Interface)
CTRL_LOAD = 0      # Stream inval[0:size[0]] into MT scratchpad at tile's spad region + d_addr[0].
CTRL_STORE = 1     # Stream size[0] elements from MT scratchpad at tile's spad region + d_addr[0] out to outval.
CTRL_PROG = 2      # Forward inval[0:size[0]*4] as program words to the target compute tile's IMEM at d_addr[0].
CTRL_RUN = 3       # Forward run command to the target compute tile. Executes program at PC = d_addr[0].
CTRL_MT_PUSH = 4   # DMA from MT scratchpad → target CT scratchpad at d_addr[0], size[0] elements.
CTRL_MT_PULL = 5   # DMA from target CT scratchpad → MT scratchpad at d_addr[0], size[0] elements.
CTRL_CT_NOP = 255  # (Internal) No-op forwarded to CT; no spad access.

@df.region()
def gemm_region(A: float32[M * K], B: float32[K * N], C: float32[M * N]):
    gemm_in_A: Stream[float32, 4][M]
    gemm_in_B: Stream[float32, 4][N]
    gemm_out_C: Stream[float32, 4][M, N]
    fifo_A: Stream[float32, 4][P0, P1]
    fifo_B: Stream[float32, 4][P0, P1]

    @df.kernel(mapping=[1], args=[A, B, C])
    def gemm_loader(A_in: float32[M * K], B_in: float32[K * N], C_in: float32[M * N]):
        for k in range(K):
            gemm_in_A[0].put(A_in[0 * K + k])
            gemm_in_A[1].put(A_in[1 * K + k])
            gemm_in_B[0].put(B_in[k * N + 0])
            gemm_in_B[1].put(B_in[k * N + 1])
        C_in[0 * N + 0] = gemm_out_C[0, 0].get()
        C_in[0 * N + 1] = gemm_out_C[0, 1].get()
        C_in[1 * N + 0] = gemm_out_C[1, 0].get()
        C_in[1 * N + 1] = gemm_out_C[1, 1].get()

    @df.kernel(mapping=[P0, P1])
    def gemm_pe():
        i, j = df.get_pid()
        # Corner cases: do nothing
        if (i == 0 or i == M + 1) and (j == 0 or j == N + 1):
            pass
        elif j == 0:
            for k in range(K):
                # Use max/min to avoid TypeInferer failing on statically dead branches
                fifo_A[i, min(P1 - 1, j + 1)].put(gemm_in_A[min(M - 1, max(0, i - 1))].get())
        elif i == 0:
            for k in range(K):
                # Use max/min to avoid TypeInferer failing on statically dead branches
                fifo_B[min(P0 - 1, i + 1), j].put(gemm_in_B[min(N - 1, max(0, j - 1))].get())
        elif i == M + 1 and j > 0:
            for k in range(K):
                _b: float32 = fifo_B[i, j].get()
        elif j == N + 1 and i > 0:
            for k in range(K):
                _a: float32 = fifo_A[i, j].get()
        else:
            c: float32 = 0.0
            for k in range(K):
                a: float32 = fifo_A[i, j].get()
                b: float32 = fifo_B[i, j].get()
                c += a * b
                fifo_A[i, min(P1 - 1, j + 1)].put(a)
                fifo_B[min(P0 - 1, i + 1), j].put(b)
            gemm_out_C[min(M - 1, max(0, i - 1)), min(N - 1, max(0, j - 1))].put(c)


@df.region()
def vadd_region(A: float32[M * K], B: float32[K * N], C: float32[M * N]):
    vadd_in_A: Stream[float32, 4][M]
    vadd_in_B: Stream[float32, 4][M]
    vadd_out_C: Stream[float32, 4][M]

    @df.kernel(mapping=[1], args=[A, B, C])
    def vadd_loader_k(A_in: float32[M * K], B_in: float32[K * N], C_in: float32[M * N]):
        vadd_in_A[0].put(A_in[0])
        vadd_in_A[1].put(A_in[1])
        vadd_in_B[0].put(B_in[0])
        vadd_in_B[1].put(B_in[1])
        C_in[0] = vadd_out_C[0].get()
        C_in[1] = vadd_out_C[1].get()

    @df.kernel(mapping=[M])
    def vadd_pe_k():
        i = df.get_pid()
        a: float32 = vadd_in_A[i].get()
        b: float32 = vadd_in_B[i].get()
        vadd_out_C[i].put(a + b)


# MT spad is partitioned into two halves:
#   Tile 0 data: mt_spad[0 .. TILE_MT_STRIDE-1]
#   Tile 1 data: mt_spad[TILE_MT_STRIDE .. 2*TILE_MT_STRIDE-1]
# CT scratchpad addresses are always relative (0 .. MEM_SIZE-1) per tile.
TILE_MT_STRIDE = 128


# ============================================================
# 2×1 Mesh with kernels
# ============================================================

@df.region()
def top_2x1(
    ctrl: int32[1],      # Scalars use int32[1] (1-element arrays) due to Allo's memref lowering.
    tile_id: int32[1],   # Access via [0].
    d_addr: int32[1],
    inval: float32[BW],
    size: int32[1],
    outval: float32[BW]
):
    # --- Streams: memory_tile ↔ compute_tiles ---
    cmd_ctrl_2x1: Stream[int32, 4][2]
    cmd_daddr_2x1: Stream[int32, 4][2]
    cmd_size_2x1: Stream[int32, 4][2]
    cmd_data_2x1: Stream[float32, 4][2]
    res_data_2x1: Stream[float32, 4][2]

    @df.kernel(mapping=[1], args=[ctrl, tile_id, d_addr, inval, size, outval])
    def mt_2x1(
        ctrl_in: int32[1],
        tile_id_in: int32[1],
        d_addr_in: int32[1],
        inval_in: float32[BW],
        size_in: int32[1],
        outval_in: float32[BW]
    ):
        mt_spad_2x1: float32[MEM_SIZE] @ Stateful = 0.0
        temp_in_2x1: float32[2, BW] @ Stateful = 0.0
        temp_out_2x1: float32[2, BW] @ Stateful = 0.0
        ct_ctrl_2x1: int32[2] @ Stateful = 0
        ct_daddr_2x1: int32[2] @ Stateful = 0
        ct_size_2x1: int32[2] @ Stateful = 0

        for t in range(2):
            ct_ctrl_2x1[t] = CTRL_CT_NOP
            ct_daddr_2x1[t] = d_addr_in[0]
            ct_size_2x1[t] = 0

        tid: int32 = tile_id_in[0]
        if tid == 0:
            if ctrl_in[0] == CTRL_LOAD:
                for i in range(size_in[0]):
                    mt_spad_2x1[d_addr_in[0] + i] = inval_in[i]
            elif ctrl_in[0] == CTRL_STORE:
                for i in range(size_in[0]):
                    outval_in[i] = mt_spad_2x1[d_addr_in[0] + i]
            elif ctrl_in[0] == CTRL_PROG:
                for i in range(size_in[0] * 4):
                    temp_in_2x1[0, i] = inval_in[i]
                ct_ctrl_2x1[0] = CTRL_PROG
                ct_size_2x1[0] = size_in[0]
            elif ctrl_in[0] == CTRL_RUN:
                ct_ctrl_2x1[0] = CTRL_RUN
            elif ctrl_in[0] == CTRL_MT_PUSH:
                for i in range(size_in[0]):
                    temp_in_2x1[0, i] = mt_spad_2x1[d_addr_in[0] + i]
                ct_ctrl_2x1[0] = CTRL_LOAD
                ct_size_2x1[0] = size_in[0]
            elif ctrl_in[0] == CTRL_MT_PULL:
                ct_ctrl_2x1[0] = CTRL_STORE
                ct_size_2x1[0] = size_in[0]

        elif tid == 1:
            if ctrl_in[0] == CTRL_LOAD:
                for i in range(size_in[0]):
                    mt_spad_2x1[TILE_MT_STRIDE + d_addr_in[0] + i] = inval_in[i]
            elif ctrl_in[0] == CTRL_STORE:
                for i in range(size_in[0]):
                    outval_in[i] = mt_spad_2x1[TILE_MT_STRIDE + d_addr_in[0] + i]
            elif ctrl_in[0] == CTRL_PROG:
                for i in range(size_in[0] * 4):
                    temp_in_2x1[1, i] = inval_in[i]
                ct_ctrl_2x1[1] = CTRL_PROG
                ct_size_2x1[1] = size_in[0]
            elif ctrl_in[0] == CTRL_RUN:
                ct_ctrl_2x1[1] = CTRL_RUN
            elif ctrl_in[0] == CTRL_MT_PUSH:
                for i in range(size_in[0]):
                    temp_in_2x1[1, i] = mt_spad_2x1[TILE_MT_STRIDE + d_addr_in[0] + i]
                ct_ctrl_2x1[1] = CTRL_LOAD
                ct_size_2x1[1] = size_in[0]
            elif ctrl_in[0] == CTRL_MT_PULL:
                ct_ctrl_2x1[1] = CTRL_STORE
                ct_size_2x1[1] = size_in[0]

        # Manual unrolling for stream arrays
        cmd_ctrl_2x1[0].put(ct_ctrl_2x1[0])
        cmd_daddr_2x1[0].put(ct_daddr_2x1[0])
        cmd_size_2x1[0].put(ct_size_2x1[0])
        for i in range(BW):
            cmd_data_2x1[0].put(temp_in_2x1[0, i])

        cmd_ctrl_2x1[1].put(ct_ctrl_2x1[1])
        cmd_daddr_2x1[1].put(ct_daddr_2x1[1])
        cmd_size_2x1[1].put(ct_size_2x1[1])
        for i in range(BW):
            cmd_data_2x1[1].put(temp_in_2x1[1, i])

        for i in range(BW):
            temp_out_2x1[0, i] = res_data_2x1[0].get()
        for i in range(BW):
            temp_out_2x1[1, i] = res_data_2x1[1].get()

        # Write CT output into MT spad for PULL
        if ctrl_in[0] == CTRL_MT_PULL:
            if tid == 0:
                for i in range(size_in[0]):
                    mt_spad_2x1[d_addr_in[0] + i] = temp_out_2x1[0, i]
            elif tid == 1:
                for i in range(size_in[0]):
                    mt_spad_2x1[TILE_MT_STRIDE + d_addr_in[0] + i] = temp_out_2x1[1, i]

    @df.kernel(mapping=[2])
    def ct_2x1():
        id = df.get_pid()
        imem: int32[IMEM_SIZE * 4] @ Stateful = 0
        data_mem: float32[MEM_SIZE] @ Stateful = 0.0
        A_buf: float32[M * K] @ Stateful = 0.0
        B_buf: float32[K * N] @ Stateful = 0.0
        C_buf: float32[M * N] @ Stateful = 0.0
        result: float32[BW] @ Stateful = 0.0
        ctrl_p: int32[1] @ Stateful = 0
        daddr_p: int32[1] @ Stateful = 0
        size_p: int32[1] @ Stateful = 0
        indata: float32[BW] @ Stateful = 0.0

        # Receive fixed command packet from memory tile
        ctrl_p[0] = cmd_ctrl_2x1[id].get()
        daddr_p[0] = cmd_daddr_2x1[id].get()
        size_p[0] = cmd_size_2x1[id].get()
        for i in range(BW):
            indata[i] = cmd_data_2x1[id].get()

        if ctrl_p[0] == CTRL_LOAD:
            for offset in range(size_p[0]):
                data_mem[daddr_p[0] + offset] = indata[offset]
        elif ctrl_p[0] == CTRL_STORE:
            for offset in range(size_p[0]):
                result[offset] = data_mem[daddr_p[0] + offset]
        elif ctrl_p[0] == CTRL_PROG:
            for i in range(size_p[0] * 4):
                imem[daddr_p[0] * 4 + i] = int(indata[i])
        elif ctrl_p[0] == CTRL_RUN:
            for pc in range(IMEM_SIZE):
                cur_op: int32 = imem[(daddr_p[0] + pc) * 4]
                if cur_op == OP_GEMM:
                    ga1: int32 = imem[(daddr_p[0] + pc) * 4 + 1]
                    ga2: int32 = imem[(daddr_p[0] + pc) * 4 + 2]
                    ga3: int32 = imem[(daddr_p[0] + pc) * 4 + 3]
                    for gi in range(M * K):
                        A_buf[gi] = data_mem[ga1 + gi]
                    for gi in range(K * N):
                        B_buf[gi] = data_mem[ga2 + gi]
                    gemm_region(A_buf, B_buf, C_buf)
                    for gi in range(M * N):
                        data_mem[ga3 + gi] = C_buf[gi]
                elif cur_op == OP_VADD:
                    va1: int32 = imem[(daddr_p[0] + pc) * 4 + 1]
                    va2: int32 = imem[(daddr_p[0] + pc) * 4 + 2]
                    va3: int32 = imem[(daddr_p[0] + pc) * 4 + 3]
                    A_buf[0] = data_mem[va1 + 0]
                    A_buf[1] = data_mem[va1 + 1]
                    B_buf[0] = data_mem[va2 + 0]
                    B_buf[1] = data_mem[va2 + 1]
                    vadd_region(A_buf, B_buf, C_buf)
                    data_mem[va3 + 0] = C_buf[0]
                    data_mem[va3 + 1] = C_buf[1]

        # Always send BW result elements back to memory tile
        for i in range(BW):
            res_data_2x1[id].put(result[i])


@df.region()
def top_2x2(
    ctrl: int32[1],      # Scalars use int32[1] (1-element arrays) due to Allo's memref lowering.
    tile_id: int32[1],   # Access via [0].
    d_addr: int32[1],
    inval: float32[BW],
    size: int32[1],
    outval: float32[BW]
):
    """
    Architecture — Kernel Topology (2x2 Mesh)
    ----------------------------------------------
    `top_2x2` is a single `df.region` containing multiple `df.kernel`s:
    - `mt_2x2_0` (mapping=[1], has host args): dispatches commands to CT_00, CT_01, and forwards to MT_1 via streams.
    - `mt_2x2_1` (mapping=[1], no host args): executes commands forwarded from MT_0 and dispatches to CT_10, CT_11.
    - `ct_2x2` (mapping=[2, 2], no args): kernels with own `@stateful` storage, calling specific `gemm_region` or `vadd_region` variants.

    Tile Addressing:
    - `tile_id[0] == 0`: CT_00 (Targeted via MT_0, offset `0`)
    - `tile_id[0] == 1`: CT_01 (Targeted via MT_0, offset `TILE_MT_STRIDE = 128`)
    - `tile_id[0] == 2`: CT_10 (Targeted via MT_1, offset `0`)
    - `tile_id[0] == 3`: CT_11 (Targeted via MT_1, offset `TILE_MT_STRIDE = 128`)
    CT scratchpad addresses are always 0-based per tile.

    Stream Protocol (Fixed-II):
    Per invocation, the memory tile and each compute tile exchange a fixed-size packet:
    - memory_tile → CT: `put(ctrl)` + `put(d_addr)` + `put(size)` + `BW×put(data)` = 3 + 16 = 19 elements
    - CT → memory_tile: `BW×put(result)` = 16 elements

    Additionally, MT_0 and MT_1 exchange fixed-size packets per invocation:
    - MT_0 → MT_1: `put(ctrl)` + `put(tile_local)` + `put(d_addr)` + `put(size)` + `BW×put(data)` = 4 + 16 = 20 elements
    - MT_1 → MT_0: `BW×put(result)` = 16 elements

    Both sides always send/receive the same counts regardless of command type, ensuring balanced streams.

    Sub-Region Dispatch:
    Compute tiles call `gemm_region` or `vadd_region` variants as sub-`df.region`s conditionally during `CTRL_RUN`. This requires:
    1. Nested OMP parallelism (`OMP_MAX_ACTIVE_LEVELS >= 4`), auto-set by the simulator backend.
    2. The `_process_function_streams` recursive fix for nested control-flow calls.
    """
    # --- Streams: memory_tiles ↔ compute_tiles ---
    cmd_ctrl_2x2: Stream[int32, 4][2, 2]
    cmd_daddr_2x2: Stream[int32, 4][2, 2]
    cmd_size_2x2: Stream[int32, 4][2, 2]
    cmd_data_2x2: Stream[float32, 4][2, 2]
    res_data_2x2: Stream[float32, 4][2, 2]

    # MT0 -> MT1 forwarding streams
    mt_fwd_ctrl: Stream[int32, 4]
    mt_fwd_tile_id: Stream[int32, 4]
    mt_fwd_daddr: Stream[int32, 4]
    mt_fwd_size: Stream[int32, 4]
    mt_fwd_data: Stream[float32, 4]
    # MT1 -> MT0 return stream
    mt_ret_data: Stream[float32, 4]

    @df.kernel(mapping=[1], args=[ctrl, tile_id, d_addr, inval, size, outval])
    def mt_2x2_0(
        ctrl_in: int32[1],
        tile_id_in: int32[1],
        d_addr_in: int32[1],
        inval_in: float32[BW],
        size_in: int32[1],
        outval_in: float32[BW]
    ):
        mt_spad_r0: float32[MEM_SIZE] @ Stateful = 0.0
        temp_in_2x2_0: float32[2, BW] @ Stateful = 0.0
        temp_out_2x2_0: float32[2, BW] @ Stateful = 0.0
        temp_fwd_in_0: float32[BW] @ Stateful = 0.0
        temp_fwd_out_0: float32[BW] @ Stateful = 0.0

        ct_ctrl_2x2_0: int32[2] @ Stateful = 0
        ct_daddr_2x2_0: int32[2] @ Stateful = 0
        ct_size_2x2_0: int32[2] @ Stateful = 0

        mt1_ctrl_2x2_0: int32[1] @ Stateful = 0
        mt1_tile_id_2x2_0: int32[1] @ Stateful = 0
        mt1_daddr_2x2_0: int32[1] @ Stateful = 0
        mt1_size_2x2_0: int32[1] @ Stateful = 0

        for t in range(2):
            ct_ctrl_2x2_0[t] = CTRL_CT_NOP
            ct_daddr_2x2_0[t] = d_addr_in[0]
            ct_size_2x2_0[t] = 0

        mt1_ctrl_2x2_0[0] = CTRL_CT_NOP
        mt1_tile_id_2x2_0[0] = 0
        mt1_daddr_2x2_0[0] = d_addr_in[0]
        mt1_size_2x2_0[0] = 0

        tid: int32 = tile_id_in[0]
        if tid == 0:
            if ctrl_in[0] == CTRL_LOAD:
                for i in range(size_in[0]):
                    mt_spad_r0[d_addr_in[0] + i] = inval_in[i]
            elif ctrl_in[0] == CTRL_STORE:
                for i in range(size_in[0]):
                    outval_in[i] = mt_spad_r0[d_addr_in[0] + i]
            elif ctrl_in[0] == CTRL_PROG:
                for i in range(size_in[0] * 4):
                    temp_in_2x2_0[0, i] = inval_in[i]
                ct_ctrl_2x2_0[0] = CTRL_PROG
                ct_size_2x2_0[0] = size_in[0]
            elif ctrl_in[0] == CTRL_RUN:
                ct_ctrl_2x2_0[0] = CTRL_RUN
            elif ctrl_in[0] == CTRL_MT_PUSH:
                for i in range(size_in[0]):
                    temp_in_2x2_0[0, i] = mt_spad_r0[d_addr_in[0] + i]
                ct_ctrl_2x2_0[0] = CTRL_LOAD
                ct_size_2x2_0[0] = size_in[0]
            elif ctrl_in[0] == CTRL_MT_PULL:
                ct_ctrl_2x2_0[0] = CTRL_STORE
                ct_size_2x2_0[0] = size_in[0]

        elif tid == 1:
            if ctrl_in[0] == CTRL_LOAD:
                for i in range(size_in[0]):
                    mt_spad_r0[TILE_MT_STRIDE + d_addr_in[0] + i] = inval_in[i]
            elif ctrl_in[0] == CTRL_STORE:
                for i in range(size_in[0]):
                    outval_in[i] = mt_spad_r0[TILE_MT_STRIDE + d_addr_in[0] + i]
            elif ctrl_in[0] == CTRL_PROG:
                for i in range(size_in[0] * 4):
                    temp_in_2x2_0[1, i] = inval_in[i]
                ct_ctrl_2x2_0[1] = CTRL_PROG
                ct_size_2x2_0[1] = size_in[0]
            elif ctrl_in[0] == CTRL_RUN:
                ct_ctrl_2x2_0[1] = CTRL_RUN
            elif ctrl_in[0] == CTRL_MT_PUSH:
                for i in range(size_in[0]):
                    temp_in_2x2_0[1, i] = mt_spad_r0[TILE_MT_STRIDE + d_addr_in[0] + i]
                ct_ctrl_2x2_0[1] = CTRL_LOAD
                ct_size_2x2_0[1] = size_in[0]
            elif ctrl_in[0] == CTRL_MT_PULL:
                ct_ctrl_2x2_0[1] = CTRL_STORE
                ct_size_2x2_0[1] = size_in[0]

        elif tid == 2 or tid == 3:
            if ctrl_in[0] == CTRL_LOAD or ctrl_in[0] == CTRL_PROG:
                for i in range(BW):
                    temp_fwd_in_0[i] = inval_in[i]
            mt1_ctrl_2x2_0[0] = ctrl_in[0]
            mt1_tile_id_2x2_0[0] = tid
            mt1_daddr_2x2_0[0] = d_addr_in[0]
            mt1_size_2x2_0[0] = size_in[0]

        # Manual unrolling for local compute tiles (0, 1)
        cmd_ctrl_2x2[0, 0].put(ct_ctrl_2x2_0[0])
        cmd_daddr_2x2[0, 0].put(ct_daddr_2x2_0[0])
        cmd_size_2x2[0, 0].put(ct_size_2x2_0[0])
        for i in range(BW):
            cmd_data_2x2[0, 0].put(temp_in_2x2_0[0, i])

        cmd_ctrl_2x2[0, 1].put(ct_ctrl_2x2_0[1])
        cmd_daddr_2x2[0, 1].put(ct_daddr_2x2_0[1])
        cmd_size_2x2[0, 1].put(ct_size_2x2_0[1])
        for i in range(BW):
            cmd_data_2x2[0, 1].put(temp_in_2x2_0[1, i])

        # Forward command to MT1
        mt_fwd_ctrl.put(mt1_ctrl_2x2_0[0])
        mt_fwd_tile_id.put(mt1_tile_id_2x2_0[0])
        mt_fwd_daddr.put(mt1_daddr_2x2_0[0])
        mt_fwd_size.put(mt1_size_2x2_0[0])
        for i in range(BW):
            mt_fwd_data.put(temp_fwd_in_0[i])

        # Receive results
        for i in range(BW):
            temp_out_2x2_0[0, i] = res_data_2x2[0, 0].get()
        for i in range(BW):
            temp_out_2x2_0[1, i] = res_data_2x2[0, 1].get()
        for i in range(BW):
            temp_fwd_out_0[i] = mt_ret_data.get()

        # Handle local PULL
        if ctrl_in[0] == CTRL_MT_PULL:
            if tid == 0:
                for i in range(size_in[0]):
                    mt_spad_r0[d_addr_in[0] + i] = temp_out_2x2_0[0, i]
            elif tid == 1:
                for i in range(size_in[0]):
                    mt_spad_r0[TILE_MT_STRIDE + d_addr_in[0] + i] = temp_out_2x2_0[1, i]
        
        # Handle remote STORE (from MT1)
        if ctrl_in[0] == CTRL_STORE and (tid == 2 or tid == 3):
            for i in range(size_in[0]):
                outval_in[i] = temp_fwd_out_0[i]

    @df.kernel(mapping=[1])
    def mt_2x2_1():
        mt_spad_r1: float32[MEM_SIZE] @ Stateful = 0.0
        temp_in_2x2_1: float32[2, BW] @ Stateful = 0.0
        temp_out_2x2_1: float32[2, BW] @ Stateful = 0.0
        temp_fwd_in_1: float32[BW] @ Stateful = 0.0
        temp_fwd_out_1: float32[BW] @ Stateful = 0.0

        ct_ctrl_2x2_1: int32[2] @ Stateful = 0
        ct_daddr_2x2_1: int32[2] @ Stateful = 0
        ct_size_2x2_1: int32[2] @ Stateful = 0

        mt1_ctrl_2x2_1: int32[1] @ Stateful = 0
        mt1_tile_id_2x2_1: int32[1] @ Stateful = 0
        mt1_daddr_2x2_1: int32[1] @ Stateful = 0
        mt1_size_2x2_1: int32[1] @ Stateful = 0
        
        for t in range(2):
            ct_ctrl_2x2_1[t] = CTRL_CT_NOP
            ct_size_2x2_1[t] = 0
        
        mt1_ctrl_2x2_1[0] = mt_fwd_ctrl.get()
        mt1_tile_id_2x2_1[0] = mt_fwd_tile_id.get()
        mt1_daddr_2x2_1[0] = mt_fwd_daddr.get()
        mt1_size_2x2_1[0] = mt_fwd_size.get()

        for t in range(2):
            ct_daddr_2x2_1[t] = mt1_daddr_2x2_1[0]

        for i in range(BW):
            temp_fwd_in_1[i] = mt_fwd_data.get()

        tid: int32 = mt1_tile_id_2x2_1[0]
        if tid == 2:
            if mt1_ctrl_2x2_1[0] == CTRL_LOAD:
                for i in range(mt1_size_2x2_1[0]):
                    mt_spad_r1[mt1_daddr_2x2_1[0] + i] = temp_fwd_in_1[i]
            elif mt1_ctrl_2x2_1[0] == CTRL_STORE:
                for i in range(mt1_size_2x2_1[0]):
                    temp_fwd_out_1[i] = mt_spad_r1[mt1_daddr_2x2_1[0] + i]
            elif mt1_ctrl_2x2_1[0] == CTRL_PROG:
                for i in range(mt1_size_2x2_1[0] * 4):
                    temp_in_2x2_1[0, i] = temp_fwd_in_1[i]
                ct_ctrl_2x2_1[0] = CTRL_PROG
                ct_size_2x2_1[0] = mt1_size_2x2_1[0]
            elif mt1_ctrl_2x2_1[0] == CTRL_RUN:
                ct_ctrl_2x2_1[0] = CTRL_RUN
            elif mt1_ctrl_2x2_1[0] == CTRL_MT_PUSH:
                for i in range(mt1_size_2x2_1[0]):
                    temp_in_2x2_1[0, i] = mt_spad_r1[mt1_daddr_2x2_1[0] + i]
                ct_ctrl_2x2_1[0] = CTRL_LOAD
                ct_size_2x2_1[0] = mt1_size_2x2_1[0]
            elif mt1_ctrl_2x2_1[0] == CTRL_MT_PULL:
                ct_ctrl_2x2_1[0] = CTRL_STORE
                ct_size_2x2_1[0] = mt1_size_2x2_1[0]

        elif tid == 3:
            if mt1_ctrl_2x2_1[0] == CTRL_LOAD:
                for i in range(mt1_size_2x2_1[0]):
                    mt_spad_r1[TILE_MT_STRIDE + mt1_daddr_2x2_1[0] + i] = temp_fwd_in_1[i]
            elif mt1_ctrl_2x2_1[0] == CTRL_STORE:
                for i in range(mt1_size_2x2_1[0]):
                    temp_fwd_out_1[i] = mt_spad_r1[TILE_MT_STRIDE + mt1_daddr_2x2_1[0] + i]
            elif mt1_ctrl_2x2_1[0] == CTRL_PROG:
                for i in range(mt1_size_2x2_1[0] * 4):
                    temp_in_2x2_1[1, i] = temp_fwd_in_1[i]
                ct_ctrl_2x2_1[1] = CTRL_PROG
                ct_size_2x2_1[1] = mt1_size_2x2_1[0]
            elif mt1_ctrl_2x2_1[0] == CTRL_RUN:
                ct_ctrl_2x2_1[1] = CTRL_RUN
            elif mt1_ctrl_2x2_1[0] == CTRL_MT_PUSH:
                for i in range(mt1_size_2x2_1[0]):
                    temp_in_2x2_1[1, i] = mt_spad_r1[TILE_MT_STRIDE + mt1_daddr_2x2_1[0] + i]
                ct_ctrl_2x2_1[1] = CTRL_LOAD
                ct_size_2x2_1[1] = mt1_size_2x2_1[0]
            elif mt1_ctrl_2x2_1[0] == CTRL_MT_PULL:
                ct_ctrl_2x2_1[1] = CTRL_STORE
                ct_size_2x2_1[1] = mt1_size_2x2_1[0]

        # Manual unrolling for compute tiles (2, 3)
        cmd_ctrl_2x2[1, 0].put(ct_ctrl_2x2_1[0])
        cmd_daddr_2x2[1, 0].put(ct_daddr_2x2_1[0])
        cmd_size_2x2[1, 0].put(ct_size_2x2_1[0])
        for i in range(BW):
            cmd_data_2x2[1, 0].put(temp_in_2x2_1[0, i])

        cmd_ctrl_2x2[1, 1].put(ct_ctrl_2x2_1[1])
        cmd_daddr_2x2[1, 1].put(ct_daddr_2x2_1[1])
        cmd_size_2x2[1, 1].put(ct_size_2x2_1[1])
        for i in range(BW):
            cmd_data_2x2[1, 1].put(temp_in_2x2_1[1, i])

        # Receive results from compute tiles (2, 3)
        for i in range(BW):
            temp_out_2x2_1[0, i] = res_data_2x2[1, 0].get()
        for i in range(BW):
            temp_out_2x2_1[1, i] = res_data_2x2[1, 1].get()

        # Handle local PULL (write CT output into MT_1 spad)
        if mt1_ctrl_2x2_1[0] == CTRL_MT_PULL:
            if tid == 2:
                for i in range(mt1_size_2x2_1[0]):
                    mt_spad_r1[mt1_daddr_2x2_1[0] + i] = temp_out_2x2_1[0, i]
            elif tid == 3:
                for i in range(mt1_size_2x2_1[0]):
                    mt_spad_r1[TILE_MT_STRIDE + mt1_daddr_2x2_1[0] + i] = temp_out_2x2_1[1, i]

        # Return results to MT0
        for i in range(BW):
            mt_ret_data.put(temp_fwd_out_1[i])


    @df.kernel(mapping=[2, 2])
    def ct_2x2():
        r, c = df.get_pid()
        imem: int32[IMEM_SIZE * 4] @ Stateful = 0
        data_mem: float32[MEM_SIZE] @ Stateful = 0.0
        A_buf: float32[M * K] @ Stateful = 0.0
        B_buf: float32[K * N] @ Stateful = 0.0
        C_buf: float32[M * N] @ Stateful = 0.0
        result: float32[BW] @ Stateful = 0.0
        ctrl_p: int32[1] @ Stateful = 0
        daddr_p: int32[1] @ Stateful = 0
        size_p: int32[1] @ Stateful = 0
        indata: float32[BW] @ Stateful = 0.0

        ctrl_p[0] = cmd_ctrl_2x2[r, c].get()
        daddr_p[0] = cmd_daddr_2x2[r, c].get()
        size_p[0] = cmd_size_2x2[r, c].get()
        for i in range(BW):
            indata[i] = cmd_data_2x2[r, c].get()

        if ctrl_p[0] == CTRL_LOAD:
            for offset in range(size_p[0]):
                data_mem[daddr_p[0] + offset] = indata[offset]
        elif ctrl_p[0] == CTRL_STORE:
            for offset in range(size_p[0]):
                result[offset] = data_mem[daddr_p[0] + offset]
        elif ctrl_p[0] == CTRL_PROG:
            for i in range(size_p[0] * 4):
                imem[daddr_p[0] * 4 + i] = int(indata[i])
        elif ctrl_p[0] == CTRL_RUN:
            for pc in range(IMEM_SIZE):
                cur_op: int32 = imem[(daddr_p[0] + pc) * 4]
                if cur_op == OP_GEMM:
                    ga1: int32 = imem[(daddr_p[0] + pc) * 4 + 1]
                    ga2: int32 = imem[(daddr_p[0] + pc) * 4 + 2]
                    ga3: int32 = imem[(daddr_p[0] + pc) * 4 + 3]
                    for gi in range(M * K):
                        A_buf[gi] = data_mem[ga1 + gi]
                    for gi in range(K * N):
                        B_buf[gi] = data_mem[ga2 + gi]
                    gemm_region(A_buf, B_buf, C_buf)
                    for gi in range(M * N):
                        data_mem[ga3 + gi] = C_buf[gi]
                elif cur_op == OP_VADD:
                    va1: int32 = imem[(daddr_p[0] + pc) * 4 + 1]
                    va2: int32 = imem[(daddr_p[0] + pc) * 4 + 2]
                    va3: int32 = imem[(daddr_p[0] + pc) * 4 + 3]
                    A_buf[0] = data_mem[va1 + 0]
                    A_buf[1] = data_mem[va1 + 1]
                    B_buf[0] = data_mem[va2 + 0]
                    B_buf[1] = data_mem[va2 + 1]
                    vadd_region(A_buf, B_buf, C_buf)
                    data_mem[va3 + 0] = C_buf[0]
                    data_mem[va3 + 1] = C_buf[1]

        for i in range(BW):
            res_data_2x2[r, c].put(result[i])


# ============================================================
# Helpers for Testing
# ============================================================

class NodeTestingDriver:
    """Helper to abstract away the low-level CTRL_ commands."""
    def __init__(self, mod, call_mod_fn):
        self.mod = mod
        self.call_mod = call_mod_fn
        self.inval = np.zeros(BW, dtype=np.float32)
        self.outval = np.zeros(BW, dtype=np.float32)

    def load_data(self, tile_id, d_addr, data):
        self.inval[:len(data)] = data
        self.call_mod(self.mod, CTRL_LOAD, tile_id, d_addr, self.inval, self.outval, len(data))

    def store_data(self, tile_id, d_addr, size):
        self.call_mod(self.mod, CTRL_STORE, tile_id, d_addr, self.inval, self.outval, size)
        return self.outval[:size].copy()

    def load_program(self, tile_id, program_ops):
        # program_ops is a list of [OP, ARGS...] lists
        prog = np.zeros(BW, dtype=np.float32)
        for i, op in enumerate(program_ops):
            prog[i*4:(i+1)*4] = op
        self.inval[:BW] = prog
        self.call_mod(self.mod, CTRL_PROG, tile_id, 0, self.inval, self.outval, len(program_ops))

    def push_to_compute_tile(self, tile_id, d_addr, size):
        self.call_mod(self.mod, CTRL_MT_PUSH, tile_id, d_addr, self.inval, self.outval, size)

    def pull_from_compute_tile(self, tile_id, d_addr, size):
        self.call_mod(self.mod, CTRL_MT_PULL, tile_id, d_addr, self.inval, self.outval, size)

    def run_compute_tile(self, tile_id, pc=0):
        self.call_mod(self.mod, CTRL_RUN, tile_id, pc, self.inval, self.outval, 0)


# Memory Addresses inside generic compute tile
ADDR_A = 0
ADDR_B = 4
ADDR_C_GEMM = 8

ADDR_VADD_A = 0
ADDR_VADD_B = 2
ADDR_VADD_C = 4

TILE_0 = 0
TILE_1 = 1
TILE_2 = 2
TILE_3 = 3

def call_mod_2x1(mod, ctrl_val, tile_id_val, d_addr_val, inval, outval, size_val):
    c = np.array([ctrl_val], dtype=np.int32)
    t = np.array([tile_id_val], dtype=np.int32)
    d = np.array([d_addr_val], dtype=np.int32)
    s = np.array([size_val], dtype=np.int32)
    mod(c, t, d, inval, s, outval)

def test_2x1():
    """2x1 mesh test. Compute Tile 0 runs VADD, Compute Tile 1 runs GEMM."""
    print("Building top_2x1 (mesh)...", flush=True)
    mod = df.build(top_2x1, target="simulator")
    print("Build complete!", flush=True)

    driver = NodeTestingDriver(mod, call_mod_2x1)

    # ---------------------------------------------------------------
    # Tile 0: VADD
    # ---------------------------------------------------------------
    print("Configuring Tile 0 for VADD...")
    A_vec = np.random.rand(2).astype(np.float32)
    B_vec = np.random.rand(2).astype(np.float32)

    # Load data into MT
    driver.load_data(TILE_0, ADDR_VADD_A, A_vec)
    driver.load_data(TILE_0, ADDR_VADD_B, B_vec)

    # Program Tile 0: VADD, NOP, VADD
    driver.load_program(TILE_0, [
        [OP_VADD, ADDR_VADD_A, ADDR_VADD_B, ADDR_VADD_C],
        [OP_NOP, 0, 0, 0],
        [OP_VADD, ADDR_VADD_A, ADDR_VADD_B, ADDR_VADD_C],
        [OP_NOP, 0, 0, 0]
    ])

    # Push data to CT, run, and pull result
    driver.push_to_compute_tile(TILE_0, ADDR_VADD_A, 2)
    driver.push_to_compute_tile(TILE_0, ADDR_VADD_B, 2)
    driver.run_compute_tile(TILE_0)
    driver.pull_from_compute_tile(TILE_0, ADDR_VADD_C, 2)

    # ---------------------------------------------------------------
    # Tile 1: GEMM
    # ---------------------------------------------------------------
    print("Configuring Tile 1 for GEMM...")
    A_mat = np.random.rand(2, 2).astype(np.float32)
    B_mat = np.random.rand(2, 2).astype(np.float32)

    # Load data into MT
    driver.load_data(TILE_1, ADDR_A, A_mat.flatten())
    driver.load_data(TILE_1, ADDR_B, B_mat.flatten())

    # Program Tile 1: GEMM, GEMM
    driver.load_program(TILE_1, [
        [OP_GEMM, ADDR_A, ADDR_B, ADDR_C_GEMM],
        [OP_GEMM, ADDR_B, ADDR_A, ADDR_C_GEMM],
        [OP_NOP, 0, 0, 0]
    ])

    # Push data to CT, run, and pull result
    driver.push_to_compute_tile(TILE_1, ADDR_A, 4)
    driver.push_to_compute_tile(TILE_1, ADDR_B, 4)
    driver.run_compute_tile(TILE_1)
    driver.pull_from_compute_tile(TILE_1, ADDR_C_GEMM, 4)

    # ---------------------------------------------------------------
    # Verify execution
    # ---------------------------------------------------------------
    res_vec = driver.store_data(TILE_0, ADDR_VADD_C, 2)
    np.testing.assert_allclose(res_vec, A_vec + B_vec, atol=1e-5)
    print(">> Tile 0 VADD Passed! <<")

    res_mat = driver.store_data(TILE_1, ADDR_C_GEMM, 4).reshape((2, 2))
    # Tile 1 executes A*B then B*A. The final result is B*A
    np.testing.assert_allclose(res_mat, np.dot(B_mat, A_mat), atol=1e-5)
    print(">> Tile 1 GEMM Passed! <<")
    print("=== 2x1 Mesh Passed ===")

def call_mod_2x2(mod, ctrl_val, tile_id_val, d_addr_val, inval, outval, size_val):
    c = np.array([ctrl_val], dtype=np.int32)
    t = np.array([tile_id_val], dtype=np.int32)
    d = np.array([d_addr_val], dtype=np.int32)
    s = np.array([size_val], dtype=np.int32)
    mod(c, t, d, inval, s, outval)

def test_2x2():
    """2x2 mesh test: 4 compute tiles, 2 memory tiles."""
    print("Building top_2x2 (2x2 mesh)...", flush=True)
    mod = df.build(top_2x2, target="vitis_hls", mode="csim", project="test_2x2.prj")
    print("Build complete!", flush=True)

    driver = NodeTestingDriver(mod, call_mod_2x2)

    # Generate random test cases
    t0_A = np.random.rand(2).astype(np.float32)
    t0_B = np.random.rand(2).astype(np.float32)

    t1_A = np.random.rand(2, 2).astype(np.float32)
    t1_B = np.random.rand(2, 2).astype(np.float32)

    t2_A = np.random.rand(2, 2).astype(np.float32)
    t2_B = np.random.rand(2, 2).astype(np.float32)

    t3_A = np.random.rand(2).astype(np.float32)
    t3_B = np.random.rand(2).astype(np.float32)

    # ---------------------------------------------------------------
    # Tile 0 (CT_00): VADD
    # ---------------------------------------------------------------
    driver.load_data(TILE_0, ADDR_VADD_A, t0_A)
    driver.load_data(TILE_0, ADDR_VADD_B, t0_B)
    driver.load_program(TILE_0, [
        [OP_VADD, ADDR_VADD_A, ADDR_VADD_B, ADDR_VADD_C],
        [OP_VADD, ADDR_VADD_A, ADDR_VADD_C, ADDR_VADD_C],
        [OP_VADD, ADDR_VADD_B, ADDR_VADD_C, ADDR_VADD_C],
        [OP_NOP, 0, 0, 0]
    ])
    driver.push_to_compute_tile(TILE_0, ADDR_VADD_A, 2)
    driver.push_to_compute_tile(TILE_0, ADDR_VADD_B, 2)

    # ---------------------------------------------------------------
    # Tile 1 (CT_01): GEMM
    # ---------------------------------------------------------------
    driver.load_data(TILE_1, ADDR_A, t1_A.flatten())
    driver.load_data(TILE_1, ADDR_B, t1_B.flatten())
    driver.load_program(TILE_1, [
        [OP_GEMM, ADDR_A, ADDR_B, ADDR_C_GEMM],
        [OP_GEMM, ADDR_A, ADDR_B, ADDR_C_GEMM],
        [OP_GEMM, ADDR_B, ADDR_A, ADDR_C_GEMM],
        [OP_NOP, 0, 0, 0]
    ])
    driver.push_to_compute_tile(TILE_1, ADDR_A, 4)
    driver.push_to_compute_tile(TILE_1, ADDR_B, 4)

    # ---------------------------------------------------------------
    # Tile 2 (CT_10): GEMM
    # ---------------------------------------------------------------
    driver.load_data(TILE_2, ADDR_A, t2_A.flatten())
    driver.load_data(TILE_2, ADDR_B, t2_B.flatten())
    driver.load_program(TILE_2, [
        [OP_GEMM, ADDR_A, ADDR_B, ADDR_C_GEMM],
        [OP_GEMM, ADDR_A, ADDR_B, ADDR_C_GEMM],
        [OP_NOP, 0, 0, 0]
    ])
    driver.push_to_compute_tile(TILE_2, ADDR_A, 4)
    driver.push_to_compute_tile(TILE_2, ADDR_B, 4)

    # ---------------------------------------------------------------
    # Tile 3 (CT_11): VADD
    # ---------------------------------------------------------------
    driver.load_data(TILE_3, ADDR_VADD_A, t3_A)
    driver.load_data(TILE_3, ADDR_VADD_B, t3_B)
    driver.load_program(TILE_3, [
        [OP_VADD, ADDR_VADD_A, ADDR_VADD_B, ADDR_VADD_C],
        [OP_VADD, ADDR_VADD_B, ADDR_VADD_A, ADDR_VADD_C],
        [OP_NOP, 0, 0, 0]
    ])
    driver.push_to_compute_tile(TILE_3, ADDR_VADD_A, 2)
    driver.push_to_compute_tile(TILE_3, ADDR_VADD_B, 2)

    # ---------------------------------------------------------------
    # Execute and sync
    # ---------------------------------------------------------------
    print("Running execution on all tiles...")
    driver.run_compute_tile(TILE_0)
    driver.run_compute_tile(TILE_1)
    driver.run_compute_tile(TILE_2)
    driver.run_compute_tile(TILE_3)

    driver.pull_from_compute_tile(TILE_0, ADDR_VADD_C, 2)
    driver.pull_from_compute_tile(TILE_1, ADDR_C_GEMM, 4)
    driver.pull_from_compute_tile(TILE_2, ADDR_C_GEMM, 4)
    driver.pull_from_compute_tile(TILE_3, ADDR_VADD_C, 2)

    # ---------------------------------------------------------------
    # Verification
    # ---------------------------------------------------------------
    # Verification for TILE_0:
    # 1. C = A + B
    # 2. C = A + C = A + (A + B) = 2A + B
    # 3. C = B + C = B + (2A + B) = 2A + 2B
    np.testing.assert_allclose(driver.store_data(TILE_0, ADDR_VADD_C, 2), 2*t0_A + 2*t0_B, atol=1e-5)
    print(">> Tile 0 (CT_00) VADD Passed! <<")

    # Tile 1 executes A*B, A*B, B*A (final C is B*A)
    np.testing.assert_allclose(driver.store_data(TILE_1, ADDR_C_GEMM, 4).reshape((2, 2)), np.dot(t1_B, t1_A), atol=1e-5)
    print(">> Tile 1 (CT_01) GEMM Passed! <<")

    np.testing.assert_allclose(driver.store_data(TILE_2, ADDR_C_GEMM, 4).reshape((2, 2)), np.dot(t2_A, t2_B), atol=1e-5)
    print(">> Tile 2 (CT_10) GEMM Passed! <<")

    # Tile 3 executes A+B, B+A (final C is B+A)
    np.testing.assert_allclose(driver.store_data(TILE_3, ADDR_VADD_C, 2), t3_B + t3_A, atol=1e-5)
    print(">> Tile 3 (CT_11) VADD Passed! <<")
    print("=== 2x2 Mesh Passed ===")

if __name__ == "__main__":
    test_2x1()
    test_2x2()
