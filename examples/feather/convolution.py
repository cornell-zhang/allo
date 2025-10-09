# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# This file implements the example of Fig. 11 in the paper
# which performs a 2D convolution and converts the layout
# from channel-last to row major at the same time.

# The FEATHER implementation in this file is generally same
# as the one in gemm.py except for parameters that describe inputs.

import argparse
from math import log2

import allo
from allo.ir.types import int8
import allo.dataflow as df
import allo.backend.hls as hls
import numpy as np


parser = argparse.ArgumentParser()
help_AH = """
PE array height of NEST. Default to 4.
"""
parser.add_argument("--AH", type=int, default=4, required=False, help=help_AH)
help_AW = """PE array width of NEST. 
This also determines size of BIRRD and must be power of 2. 
Each value needs a corresponding size of BIRRD instruction. 
Currently the only available choice is 4."""
parser.add_argument(
    "--AW", type=int, default=4, choices=[4], required=False, help=help_AW
)
help_N = """Number of input activations. Default to 1."""
parser.add_argument("--N", type=int, default=1, required=False, help=help_N)
help_C = """Number of channels. Requirement: C % AW == 0. Default to 8."""
parser.add_argument("--C", type=int, default=8, required=False, help=help_C)
help_H = (
    """H dimension of input activation. Requirement: P * Q % AW == 0. Default to 8."""
)
parser.add_argument("--H", type=int, default=8, required=False, help=help_H)
help_W = (
    """W dimension of input activation. Requirement: P * Q % AW == 0. Default to 7."""
)
parser.add_argument("--W", type=int, default=7, required=False, help=help_W)
help_M = """Number of kernels (weights). Requirement: M % AH == 0. Default to 4."""
parser.add_argument("--M", type=int, default=4, required=False, help=help_M)
help_R = (
    """R dimension of convolution kernel. Requirement: P * Q % AW == 0. Default to 4."""
)
parser.add_argument("--R", type=int, default=4, required=False, help=help_R)
help_S = (
    """S dimension of convolution kernel. Requirement: P * Q % AW == 0. Default to 4."""
)
parser.add_argument("--S", type=int, default=4, required=False, help=help_S)

args = parser.parse_args()

Ty = int8
AH = args.AH
AW = args.AW

# Data parameters
N, C, H, W, M, R, S = args.N, args.C, args.H, args.W, args.M, args.R, args.S
P, Q = H - R + 1, W - S + 1

# Tile size parameters (NEST parameters)
# Dimension N is always the outmost loop
Mt, Ct, VN_size = AH, AW, AH
# Sliding windows are tiled with AW * AH size

# BIRRD parameters
LOG2_AW = int(log2(AW))
P0 = 2 * LOG2_AW if AW > 4 else 2 * LOG2_AW - 1  # Number of stages
P1 = AW // 2  # Number of switches in a stage

PS = 0  # Pass
AR = 1  # Add Right
AL = 2  # Add Left
SW = 3  # Swap


def reverse_bits(data: int, bit_range: int) -> int:
    mask = (1 << bit_range) - 1
    reversed_bits = 0
    for i in range(0, bit_range):
        if data & (1 << i):
            reversed_bits |= 1 << (bit_range - 1 - i)
    return (data & ~mask) | reversed_bits


@df.region()
def top():
    nest_out = df.pipe(dtype=Ty, shape=(AW,), depth=AH)

    @df.kernel(mapping=[1])
    def NEST(iActs: Ty[VN_size, Ct], weights: Ty[Mt, Ct, VN_size]):
        # With padding, Qt == Wt, Rt == Ht
        for m in range(AH):  # Rows, can be pipelined
            local_result: Ty[AW] = 0
            for c in range(AW):  # Cols, can be fully parallelized
                for vn in range(VN_size):  # Iterations of a local reduction
                    last_res: Ty = 0.0 if vn == 0 else local_result[c]
                    iAct: Ty = iActs[vn, c]
                    weight: Ty = weights[m, c, vn]
                    result: Ty = last_res + iAct * weight
                    local_result[c] = result
            nest_out.put(local_result)

    connection = df.array(df.pipe(dtype=Ty, shape=(), depth=1), shape=(P0 + 1, P1 * 2))

    @df.kernel(mapping=[1])
    def bus():
        for _ in range(AH):
            array: Ty[AW] = nest_out.get()
            with allo.meta_for(AW) as i:
                connection[0, i].put(array[i])

    inst_input = df.array(df.pipe(dtype=int8, shape=(), depth=1), shape=(P0, P1))

    @df.kernel(mapping=[1])
    def inst_rw(insts: int8[P0, P1]):
        with allo.meta_for(P0) as i:
            with allo.meta_for(P1) as j:
                inst_input[i, j].put(insts[i, j])

    @df.kernel(mapping=[P0, P1])
    def BIRRD():
        i, j = df.get_pid()
        inst = inst_input[i, j].get()  # Update inst every cycle
        for _ in range(AH):
            # The first stage
            with allo.meta_if(i == 0):
                in_left: Ty = connection[0, 2 * j].get()
                in_right: Ty = connection[0, 2 * j + 1].get()
                out_left: Ty = 0.0
                out_right: Ty = 0.0
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
                in_left: Ty = connection[P0 - 1, 2 * j].get()
                in_right: Ty = connection[P0 - 1, 2 * j + 1].get()
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
                in_left: Ty = connection[i, 2 * j].get()
                in_right: Ty = connection[i, 2 * j + 1].get()
                out_left: Ty = 0.0
                out_right: Ty = 0.0
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
    def output(output_buffer: Ty[AH, AW]):
        for d in range(AH):
            with allo.meta_for(AW) as i:
                res = connection[P0, i].get()
                output_buffer[d, i] += res


def convolve_2d_row_major(mats: np.ndarray, kernels: np.ndarray) -> np.ndarray:
    N, C, H, W = mats.shape
    M, Ck, R, S = kernels.shape
    assert C == Ck
    P, Q = H - R + 1, W - S + 1
    result = np.zeros((N, M, P, Q), dtype=np.int8)
    for n in range(N):
        for c in range(C):
            for p in range(P):
                for q in range(Q):
                    region = mats[n, c, p : p + R, q : q + S]
                    for m in range(M):
                        result[n, m, p, q] += np.sum(region * kernels[m, c, :, :])
    return result


def test_FEATHER_conv():
    if AW == 4:
        inst0 = np.array([[AL, AL], [AL, PS], [PS, PS]]).astype(np.int8)
        inst1 = np.array([[AL, AL], [AL, PS], [SW, PS]]).astype(np.int8)
        inst2 = np.array([[AL, AL], [AR, PS], [PS, PS]]).astype(np.int8)
        inst3 = np.array([[AL, AL], [AR, PS], [PS, SW]]).astype(np.int8)
        insts = inst0, inst1, inst2, inst3
    iActs = np.random.randint(low=0, high=127, size=(N, C, H, W), dtype=np.int8)
    # iActs = np.random.rand(N, C, H, W).astype(np.float32)
    iActs_channel_last = np.ascontiguousarray(
        iActs.transpose(0, 2, 3, 1)
    )  # NCHW -> NHWC
    weights = np.random.randint(low=0, high=127, size=(M, C, R, S), dtype=np.int8)
    # weights = np.random.rand(M, C, R, S).astype(np.float32)
    weights_flattened = weights.reshape(M, C, R * S)
    oActs_row_major = np.zeros((M * P * Q // AW, AW), dtype=np.int8)

    sim_mod = df.build(top, target="simulator")
    # Outer loop: for all sliding windows
    for nt in range(0, N, 1):
        for pt in range(0, P, 1):
            for qt in range(0, Q, 1):
                iActs_sliding_window = iActs_channel_last[
                    nt, pt : pt + R, qt : qt + S, :
                ].reshape(R * S, C)
                # Inner loop: for all tiles
                for mt in range(0, M, Mt):
                    output_buffer = np.zeros((AH, AW), dtype=np.int8)
                    line_offset = (
                        pt * Q + qt
                    ) // AW  # The actual output line where (pt, qt) is
                    intraline_offset = (
                        pt * Q + qt
                    ) % AW  # The position of (pt, qt) within this line
                    for ct in range(0, C, Ct):
                        for vn in range(0, R * S, VN_size):
                            iActs_tile = np.ascontiguousarray(
                                iActs_sliding_window[vn : vn + VN_size, ct : ct + Ct]
                            )
                            weights_tile = np.ascontiguousarray(
                                weights_flattened[
                                    mt : mt + Mt, ct : ct + Ct, vn : vn + VN_size
                                ]
                            )
                            inst = insts[intraline_offset]
                            sim_mod(iActs_tile, weights_tile, inst, output_buffer)
                    for m in range(mt, mt + Mt):
                        oActs_row_major[
                            m * P * Q // AW + line_offset, intraline_offset
                        ] = output_buffer[m - mt, intraline_offset]

    ref = convolve_2d_row_major(iActs, weights)
    np.testing.assert_allclose(ref.flatten(), oActs_row_major.flatten())
    print("Dataflow Simulator Passed!")


if __name__ == "__main__":
    test_FEATHER_conv()
