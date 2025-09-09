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
PE array height of NEST. This also controls sizes of iActs and weight for simplicity. Default to 8.
"""
parser.add_argument("--AH", type=int, default=8, required=False, help=help_AH)
help_AW = """PE array width of NEST. 
This also determines size of BIRRD and must be power of 2. 
Each value needs a corresponding size of BIRRD instruction array. 
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

# M % (AW/2) == 0 to ensure unified bank depth
# K % 2 == 0  global spacial reduction
# N % AH == 0
M, K, N = 16, 16, 16
Mt, Nt = M // P1, AH  # Tile size

ID = M * K // AW  # Bank depth of iActs matrix in memory,
OD = M // P1 * N  # Bank depth of oActs matrix in memory, = AH * ID


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
    def NEST(iActs: float32[AW, ID], weights: float32[K, N]):
        for n in range(N // Nt):  # Number of N tiles, reuse over time
            for m in range(Mt):  # Tile size of M dimension
                for i in range(AH):  # Rows, can be pipelined
                    local_result: float32[AW] = 0
                    for j in range(AW):  # Cols, can be fully parallelized
                        for k in range(K // 2):  # Iterations of a local reduction
                            last_res: float32 = 0.0 if k == 0 else local_result[j]
                            iAct: float32 = iActs[
                                j, m * (K // 2) + k
                            ]  # TODO: fix this?
                            weight: float32 = (
                                weights[k, n * Nt + i]
                                if j < AW // 2
                                else weights[K // 2 + k, n * Nt + i]
                            )
                            result: float32 = last_res + iAct * weight
                            local_result[j] = result
                    nest_out.put(local_result)

    connection = df.array(
        df.pipe(dtype=float32, shape=(), depth=1), shape=(P0 + 1, P1 * 2)
    )

    @df.kernel(mapping=[1])
    def bus():
        for _ in range(OD):
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
        for _ in range(OD):
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
    def output(oActs: float32[AW, OD]):
        for d in range(OD):
            with allo.meta_for(AW) as i:
                oActs[i, d] = connection[P0, i].get()


def iAct_tiling(iActs: np.ndarray):
    # Split the array vertically into 2 parts (prepare for K reduction)
    B_left, B_right = np.hsplit(iActs, 2)
    # M tiling: each sub array corresponds to an input bank (a PE column)
    C_left = np.vsplit(B_left, AW // 2)
    C_right = np.vsplit(B_right, AW // 2)
    # Flatten data in one bank
    # And combine the rows (banks) into the array as the stationary input
    D = np.vstack([C.flatten() for C in C_left + C_right])
    return np.ascontiguousarray(D)


def oAct_make_layout(oActs_raw: np.ndarray):
    # Divide on N first
    B = np.hsplit(oActs_raw, N // AH)
    C = map(lambda b: b.flatten().reshape(P1, -1), B)
    D = np.concatenate(list(C), axis=1)
    return np.ascontiguousarray(D)


def correctness_check(oActs, ref):
    if AW == 16:
        np.testing.assert_allclose(ref[0], oActs[8], atol=1e-5)
        np.testing.assert_allclose(ref[1], oActs[10], atol=1e-5)
        np.testing.assert_allclose(ref[2], oActs[11], atol=1e-5)
        np.testing.assert_allclose(ref[3], oActs[9], atol=1e-5)
        np.testing.assert_allclose(ref[4], oActs[5], atol=1e-5)
        np.testing.assert_allclose(ref[5], oActs[6], atol=1e-5)
        np.testing.assert_allclose(ref[6], oActs[7], atol=1e-5)
        np.testing.assert_allclose(ref[7], oActs[4], atol=1e-5)
    elif AW == 8:
        np.testing.assert_allclose(ref[0], oActs[6], atol=1e-5)
        np.testing.assert_allclose(ref[1], oActs[5], atol=1e-5)
        np.testing.assert_allclose(ref[2], oActs[2], atol=1e-5)
        np.testing.assert_allclose(ref[3], oActs[1], atol=1e-5)
    elif AW == 4:
        np.testing.assert_allclose(ref[0], oActs[2], atol=1e-5)
        np.testing.assert_allclose(ref[1], oActs[0], atol=1e-5)


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
    iActs = iAct_tiling(iActs_no_layout)
    weights = np.random.rand(K, N).astype(np.float32)

    sim_mod = df.build(top, target="simulator")
    oActs = np.zeros((AW, OD), dtype=np.float32)
    sim_mod(iActs, weights, inst, oActs)
    oActs_no_layout = np.dot(iActs_no_layout, weights)  # MxN
    ref = oAct_make_layout(oActs_no_layout)
    correctness_check(oActs, ref)
    print("Dataflow Simulator Passed!")

    if hls.is_available("vitis_hls"):
        csim_mod = df.build(top)
        oActs = np.zeros((AW, OD), dtype=np.float32)
        csim_mod(iActs, weights, inst, oActs)
        correctness_check(oActs, ref)
        print("HLS CSIM passed!")


if __name__ == "__main__":
    test_FEATHER_GEMM()
