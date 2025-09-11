# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# FEATHER is an accelerator proposed in paper:
# Jianming Tong, Anirudh Itagi, Prasanth Chatarasi, Tushar Krishna,
# "FEATHER: A Reconfigurable Accelerator with Data Reordering
# Support for Low-Cost On-Chip Dataflow Switching", 2024 ACM/IEEE 51st
# Annual International Symposium on Computer Architecture (ISCA).

# Paper link: https://arxiv.org/abs/2405.13170
# Original RTL code: https://github.com/maeri-project/FEATHER

# This file implements the 2nd GEMM example of Fig. 10 of the paper.
# This implementation doesn't contain the quantization module
# and ping-pong buffers presented in the original paper.

from math import log2
import argparse

import allo
from allo.ir.types import float32, int8
import allo.dataflow as df
import allo.backend.hls as hls
import numpy as np

parser = argparse.ArgumentParser()
help_AH = """
PE array height of NEST. Default to 8.
"""
parser.add_argument("--AH", type=int, default=8, required=False, help=help_AH)
help_AW = """PE array width of NEST. 
This also determines size of BIRRD and must be power of 2. 
Each value needs a corresponding size of BIRRD instruction. 
Choose within [4, 8, 16] (preset instruction provided). Default to 8."""
parser.add_argument(
    "--AW", type=int, default=8, choices=[4, 8, 16], required=False, help=help_AW
)

args = parser.parse_args()

AH = args.AH
AW = args.AW

# BIRRD parameters
LOG2_AW = int(log2(AW))
P0 = 2 * LOG2_AW if AW > 4 else 2 * LOG2_AW - 1  # Number of stages
P1 = AW // 2  # Number of switches in a stage

PS = 0  # Pass
AR = 1  # Add Right
AL = 2  # Add Left
SW = 3  # Swap

# Tile size, irrelevant with input size
# Requirement: Kt % 2 == 0
Mt, Nt, Kt = AW // 2, AH, 8

# Requirements:
# M % Mt == 0, N % Nt == 0, K % Kt == 0
M, N, K = 16, 32, 16


def reverse_bits(data: int, bit_range: int) -> int:
    mask = (1 << bit_range) - 1
    reversed_bits = 0
    for i in range(0, bit_range):
        if data & (1 << i):
            reversed_bits |= 1 << (bit_range - 1 - i)
    return (data & ~mask) | reversed_bits


@df.region()
def top():
    nest_out = df.pipe(dtype=float32, shape=(AW,), depth=AH)

    @df.kernel(mapping=[1])
    def NEST(iActs: float32[Mt * 2, Kt // 2], weights: float32[Kt, Nt]):
        for i in range(AH):  # Rows, can be pipelined
            local_result: float32[AW] = 0
            for j in range(AW):  # Cols, can be fully parallelized
                for k in range(Kt // 2):  # Iterations of a local reduction
                    last_res: float32 = 0.0 if k == 0 else local_result[j]
                    iAct: float32 = iActs[j, k]
                    weight: float32 = (
                        weights[k, i] if j < AW // 2 else weights[Kt // 2 + k, i]
                    )
                    result: float32 = last_res + iAct * weight
                    local_result[j] = result
            nest_out.put(local_result)

    connection = df.array(
        df.pipe(dtype=float32, shape=(), depth=1), shape=(P0 + 1, P1 * 2)
    )

    @df.kernel(mapping=[1])
    def bus():
        for _ in range(Nt):
            array: float32[AW] = nest_out.get()
            with allo.meta_for(AW) as i:
                connection[0, i].put(array[i])

    inst_input = df.array(df.pipe(dtype=int8, shape=(), depth=1), shape=(P0, P1))

    @df.kernel(mapping=[1])
    def inst_rw(inst: int8[P0, P1]):
        with allo.meta_for(P0) as i:
            with allo.meta_for(P1) as j:
                inst_input[i, j].put(inst[i, j])

    @df.kernel(mapping=[P0, P1])
    def BIRRD():
        i, j = df.get_pid()
        inst = inst_input[i, j].get()
        for _ in range(AH):
            # The first stage
            with allo.meta_if(i == 0):
                in_left: float32 = connection[0, 2 * j].get()
                in_right: float32 = connection[0, 2 * j + 1].get()
                out_left: float32 = 0.0
                out_right: float32 = 0.0
                if inst == 0:  # Pass
                    out_left = in_left
                    out_right = in_right
                elif inst == 1:  # Add Right
                    out_left = in_left
                    out_right = in_left + in_right
                elif inst == 2:  # Add Left
                    out_left = in_left + in_right
                    out_right = in_right
                else:  # Swap
                    out_left = in_right
                    out_right = in_left
                connection[i + 1, reverse_bits(2 * j, 2)].put(out_left)
                connection[i + 1, reverse_bits(2 * j + 1, 2)].put(out_right)

            # The last stage
            with allo.meta_elif(i == P0 - 1):
                in_left: float32 = connection[P0 - 1, 2 * j].get()
                in_right: float32 = connection[P0 - 1, 2 * j + 1].get()
                if inst == 0:  # Pass
                    connection[P0, 2 * j].put(in_left)
                    connection[P0, 2 * j + 1].put(in_right)
                elif inst == 1:  # Add Right
                    connection[P0, 2 * j].put(in_left)
                    connection[P0, 2 * j + 1].put(in_left + in_right)
                elif inst == 2:  # Add Left
                    connection[P0, 2 * j].put(in_left + in_right)
                    connection[P0, 2 * j + 1].put(in_right)
                else:  # Swap
                    connection[P0, 2 * j].put(in_right)
                    connection[P0, 2 * j + 1].put(in_left)

            # Stages in the middle
            with allo.meta_else():
                in_left: float32 = connection[i, 2 * j].get()
                in_right: float32 = connection[i, 2 * j + 1].get()
                out_left: float32 = 0.0
                out_right: float32 = 0.0
                if inst == 0:  # Pass
                    out_left = in_left
                    out_right = in_right
                elif inst == 1:  # Add Right
                    out_left = in_left
                    out_right = in_left + in_right
                elif inst == 2:  # Add Left
                    out_left = in_left + in_right
                    out_right = in_right
                else:  # Swap
                    out_left = in_right
                    out_right = in_left
                connection[
                    i + 1, reverse_bits(2 * j, min(LOG2_AW, 2 + i, 2 * LOG2_AW - i))
                ].put(out_left)
                connection[
                    i + 1, reverse_bits(2 * j + 1, min(LOG2_AW, 2 + i, 2 * LOG2_AW - i))
                ].put(out_right)

    @df.kernel(mapping=[1])
    def output(output_buffer: float32[AW, AH]):
        for d in range(AH):
            with allo.meta_for(AW) as i:
                output_buffer[i, d] += connection[P0, i].get()


def iAct_tile_reorder(tile: np.ndarray):
    assert tile.shape[0] == Mt and tile.shape[1] == Kt
    # Split on K dimension for later reduction
    B_left, B_right = np.hsplit(tile, 2)
    C_left, C_right = np.vsplit(B_left, 2), np.vsplit(B_right, 2)
    D = np.vstack([C.flatten() for C in C_left + C_right])
    assert D.shape[1] == AW and D.shape[0] == Kt // 2
    return np.ascontiguousarray(D)


def oAct_make_layout(oActs_raw: np.ndarray):
    # Divide on N first
    B = np.hsplit(oActs_raw, N // AH)
    C = map(lambda b: b.flatten().reshape(P1, -1), B)
    D = np.concatenate(list(C), axis=1)
    return np.ascontiguousarray(D)


def correctness_check(oActs: np.ndarray, ref: np.ndarray):
    for m in range(M // Mt):
        if AW == 16:  # Mt == 8
            np.testing.assert_allclose(ref[m * Mt], oActs[m * 2 * Mt + 8], atol=1e-5)
            np.testing.assert_allclose(
                ref[m * Mt + 1], oActs[m * 2 * Mt + 10], atol=1e-5
            )
            np.testing.assert_allclose(
                ref[m * Mt + 2], oActs[m * 2 * Mt + 11], atol=1e-5
            )
            np.testing.assert_allclose(
                ref[m * Mt + 3], oActs[m * 2 * Mt + 9], atol=1e-5
            )
            np.testing.assert_allclose(
                ref[m * Mt + 4], oActs[m * 2 * Mt + 5], atol=1e-5
            )
            np.testing.assert_allclose(
                ref[m * Mt + 5], oActs[m * 2 * Mt + 6], atol=1e-5
            )
            np.testing.assert_allclose(
                ref[m * Mt + 6], oActs[m * 2 * Mt + 7], atol=1e-5
            )
            np.testing.assert_allclose(
                ref[m * Mt + 7], oActs[m * 2 * Mt + 4], atol=1e-5
            )
        elif AW == 8:  # Mt == 4
            np.testing.assert_allclose(ref[m * Mt], oActs[m * 2 * Mt + 6], atol=1e-5)
            np.testing.assert_allclose(
                ref[m * Mt + 1], oActs[m * 2 * Mt + 5], atol=1e-5
            )
            np.testing.assert_allclose(
                ref[m * Mt + 2], oActs[m * 2 * Mt + 2], atol=1e-5
            )
            np.testing.assert_allclose(
                ref[m * Mt + 3], oActs[m * 2 * Mt + 1], atol=1e-5
            )
        elif AW == 4:
            np.testing.assert_allclose(
                ref[m * Mt + 0], oActs[m * 2 * Mt + 2], atol=1e-5
            )
            np.testing.assert_allclose(
                ref[m * Mt + 1], oActs[m * 2 * Mt + 0], atol=1e-5
            )


def test_FEATHER_GEMM():
    if AW == 16:
        inst = np.array(
            [
                [PS, SW, PS, SW, PS, SW, PS, SW],
                [PS, PS, SW, PS, PS, PS, SW, PS],
                [PS, PS, PS, PS, PS, PS, PS, PS],
                [AL, AL, AL, AL, AR, AR, AR, AR],
                [SW, SW, SW, SW, SW, SW, SW, SW],
                [PS, PS, PS, PS, PS, PS, PS, PS],
                [PS, PS, PS, PS, PS, PS, PS, PS],
                [PS, PS, PS, PS, PS, PS, PS, PS],
            ],
            dtype=np.int8,
        )
    elif AW == 8:
        inst = np.array(
            [
                [PS, PS, PS, PS],
                [PS, PS, PS, PS],
                [AR, AR, AL, AL],
                [SW, SW, SW, SW],
                [SW, PS, PS, SW],
                [PS, PS, PS, PS],
            ],
            dtype=np.int8,
        )
    elif AW == 4:
        inst = np.array([[PS, PS], [AR, AL], [SW, PS]], dtype=np.int8)

    iActs_no_layout = np.random.rand(M, K).astype(np.float32)
    weights = np.random.rand(K, N).astype(np.float32)
    oActs = np.zeros((2 * M, N), dtype=np.float32)

    sim_mod = df.build(top, target="simulator")
    for n in range(N // Nt):
        for m in range(M // Mt):
            output_buffer = np.zeros((AW, Nt), dtype=np.float32)
            for k in range(K // Kt):
                weights_tile = np.ascontiguousarray(
                    weights[k * Kt : (k + 1) * Kt, n * Nt : (n + 1) * Nt]
                )
                iActs_tile_no_layout = iActs_no_layout[
                    m * Mt : (m + 1) * Mt, k * Kt : (k + 1) * Kt
                ]
                iActs_tile = iAct_tile_reorder(iActs_tile_no_layout)
                sim_mod(iActs_tile, weights_tile, inst, output_buffer)
            np.copyto(
                dst=oActs[m * 2 * Mt : (m + 1) * 2 * Mt, n * Nt : (n + 1) * Nt],
                src=output_buffer,
            )

    ref = np.dot(iActs_no_layout, weights)  # MxN
    correctness_check(oActs, ref)
    print("Dataflow Simulator Passed!")

    if hls.is_available("vitis_hls"):
        csim_mod = df.build(top)
        oActs = np.zeros((2 * M, N), dtype=np.float32)
        for n in range(N // Nt):
            for m in range(M // Mt):
                output_buffer = np.zeros((AW, Nt), dtype=np.float32)
                for k in range(K // Kt):
                    weights_tile = np.ascontiguousarray(
                        weights[k * Kt : (k + 1) * Kt, n * Nt : (n + 1) * Nt]
                    )
                    iActs_tile_no_layout = iActs_no_layout[
                        m * Mt : (m + 1) * Mt, k * Kt : (k + 1) * Kt
                    ]
                    iActs_tile = iAct_tile_reorder(iActs_tile_no_layout)
                    csim_mod(iActs_tile, weights_tile, inst, output_buffer)
                np.copyto(
                    dst=oActs[m * 2 * Mt : (m + 1) * 2 * Mt, n * Nt : (n + 1) * Nt],
                    src=output_buffer,
                )
        correctness_check(oActs, ref)
        print("HLS CSIM passed!")


if __name__ == "__main__":
    test_FEATHER_GEMM()
