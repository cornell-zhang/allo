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
import os

import allo
from allo.ir.types import int8, UInt
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
help_M = """M dimension (rows of A/output). Requirement: M % Mt == 0. Default to 16."""
help_N = """N dimension (cols of B/output). Requirement: N % Nt == 0. Default to 32."""
help_K = """K dimension (shared inner). Requirement: K % Kt == 0. Default to 16."""
parser.add_argument("--M", type=int, default=16, required=False, help=help_M)
parser.add_argument("--N", type=int, default=32, required=False, help=help_N)
parser.add_argument("--K", type=int, default=16, required=False, help=help_K)
# AH=AW=16, MNK=8x16x32

args = parser.parse_args()

Ty = int8
P_FACTOR = 16
TyPacked = UInt(Ty.bits * P_FACTOR)
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
Mt, Nt, Kt = AW // 2, AH, 8
M, N, K = args.M, args.N, args.K
assert K % Kt == 0
assert N % Nt == 0
assert M % Mt == 0
assert Kt % 2 == 0
assert K >= Kt
assert N >= Nt
assert M >= Mt


def reverse_bits(data: int, bit_range: int) -> int:
    mask = (1 << bit_range) - 1
    reversed_bits = 0
    for i in range(0, bit_range):
        if data & (1 << i):
            reversed_bits |= 1 << (bit_range - 1 - i)
    return (data & ~mask) | reversed_bits


@df.region()
def top():
    nest_out = df.pipe(dtype=TyPacked, depth=AH)

    @df.kernel(mapping=[1])
    def NEST(iActs: Ty[Mt * 2, Kt // 2], weights: Ty[Kt, Nt]):
        for i in allo.grid(AH, name="nest"):  # Rows, can be pipelined
            local_result: TyPacked = 0
            for j in range(AW):  # Cols, can be fully parallelized
                # FIXME: This multiplication op requires a DSP, leading to II=2
                start: int8 = j * P_FACTOR
                end: int8 = start + P_FACTOR
                for k in range(Kt // 2):  # Iterations of a local reduction
                    iAct: Ty = iActs[j, k]
                    weight: Ty = (
                        weights[k, i] if j < AW // 2 else weights[Kt // 2 + k, i]
                    )
                    local_result[start:end] += iAct * weight
            nest_out.put(local_result)

    connection = df.array(df.pipe(dtype=Ty, shape=(), depth=1), shape=(P0 + 1, P1 * 2))

    @df.kernel(mapping=[1])
    def bus():
        for _ in range(Nt):
            array: TyPacked = nest_out.get()
            with allo.meta_for(AW) as i:
                connection[0, i].put(array[i * P_FACTOR : (i + 1) * P_FACTOR])

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
    def output(output_buffer: Ty[AW, AH]):
        for d in range(AH):
            with allo.meta_for(AW) as i:
                # TODO: Do reduction on-chip
                output_buffer[i, d] = connection[P0, i].get()


def iAct_tile_reorder(tile: np.ndarray):
    assert tile.shape[0] == Mt and tile.shape[1] == Kt
    # Split on K dimension for later reduction
    B_left, B_right = np.hsplit(tile, 2)
    C_left, C_right = np.vsplit(B_left, 2), np.vsplit(B_right, 2)
    D = np.vstack([C.flatten() for C in C_left + C_right])
    # assert D.shape[1] == AW and D.shape[0] == Kt // 2
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
    print("=" * 80)
    print(f"FEATHER GEMM Test Configuration")
    print("=" * 80)
    print(f"Array Height (AH): {AH}")
    print(f"Array Width (AW): {AW}")
    print(f"Matrix dimensions - M: {M}, K: {K}, N: {N}")
    print(f"Tile dimensions - Mt: {Mt}, Kt: {Kt}, Nt: {Nt}")
    print(f"Number of stages (P0): {P0}")
    print(f"Switches per stage (P1): {P1}")
    print("-" * 80)

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

    print(f"Instruction array shape: {inst.shape}")
    print(f"Instruction array:\n{inst}")
    print("-" * 80)

    iActs_no_layout = np.random.randint(-4, 4, size=(M, K)).astype(np.int8)
    weights = np.random.randint(-4, 4, size=(K, N)).astype(np.int8)
    oActs = np.zeros((2 * M, N), dtype=np.int8)

    print(f"Input activations shape: {iActs_no_layout.shape}")
    print(f"Weights shape: {weights.shape}")
    print(f"Output activations shape (allocated): {oActs.shape}")
    print("-" * 80)

    print("Running Dataflow Simulator...")
    os.environ["OMP_NUM_THREADS"] = "256"
    sim_mod = df.build(top, target="simulator")
    print(f"# of tiles: {N // Nt}, {M // Mt}, {K // Kt}")
    for n in range(N // Nt):
        for m in range(M // Mt):
            output_buffer = np.zeros((AW, Nt), dtype=np.int8)
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
    print(f"Reference computation completed.")
    print("-" * 80)

    print("Verifying Results:")
    test_passed = True

    try:
        correctness_check(oActs, ref)
        print("🎉 Dataflow Simulator: ALL TESTS PASSED!")

        # Detailed verification for debugging
        if AW == 16:  # Mt == 8
            mapping = [(0, 8), (1, 10), (2, 11), (3, 9), (4, 5), (5, 6), (6, 7), (7, 4)]
            for m in range(M // Mt):
                for ref_idx, oact_idx in mapping:
                    actual_ref_idx = m * Mt + ref_idx
                    actual_oact_idx = m * 2 * Mt + oact_idx
                    max_diff = np.max(
                        np.abs(ref[actual_ref_idx] - oActs[actual_oact_idx])
                    )
                    print(
                        f"  ✓ ref[{actual_ref_idx}] vs oActs[{actual_oact_idx}]: PASS (max diff: {max_diff:.2e})"
                    )
        elif AW == 8:  # Mt == 4
            mapping = [(0, 6), (1, 5), (2, 2), (3, 1)]
            for m in range(M // Mt):
                for ref_idx, oact_idx in mapping:
                    actual_ref_idx = m * Mt + ref_idx
                    actual_oact_idx = m * 2 * Mt + oact_idx
                    max_diff = np.max(
                        np.abs(ref[actual_ref_idx] - oActs[actual_oact_idx])
                    )
                    print(
                        f"  ✓ ref[{actual_ref_idx}] vs oActs[{actual_oact_idx}]: PASS (max diff: {max_diff:.2e})"
                    )
        elif AW == 4:
            mapping = [(0, 2), (1, 0)]
            for m in range(M // Mt):
                for ref_idx, oact_idx in mapping:
                    actual_ref_idx = m * Mt + ref_idx
                    actual_oact_idx = m * 2 * Mt + oact_idx
                    max_diff = np.max(
                        np.abs(ref[actual_ref_idx] - oActs[actual_oact_idx])
                    )
                    print(
                        f"  ✓ ref[{actual_ref_idx}] vs oActs[{actual_oact_idx}]: PASS (max diff: {max_diff:.2e})"
                    )

    except AssertionError as e:
        print("❌ Dataflow Simulator: SOME TESTS FAILED!")
        test_passed = False

        # Detailed failure analysis
        if AW == 16:  # Mt == 8
            mapping = [(0, 8), (1, 10), (2, 11), (3, 9), (4, 5), (5, 6), (6, 7), (7, 4)]
            for m in range(M // Mt):
                for ref_idx, oact_idx in mapping:
                    actual_ref_idx = m * Mt + ref_idx
                    actual_oact_idx = m * 2 * Mt + oact_idx
                    try:
                        np.testing.assert_allclose(
                            ref[actual_ref_idx], oActs[actual_oact_idx], atol=1e-5
                        )
                        max_diff = np.max(
                            np.abs(ref[actual_ref_idx] - oActs[actual_oact_idx])
                        )
                        print(
                            f"  ✓ ref[{actual_ref_idx}] vs oActs[{actual_oact_idx}]: PASS (max diff: {max_diff:.2e})"
                        )
                    except AssertionError:
                        max_diff = np.max(
                            np.abs(ref[actual_ref_idx] - oActs[actual_oact_idx])
                        )
                        print(
                            f"  ✗ ref[{actual_ref_idx}] vs oActs[{actual_oact_idx}]: FAIL (max diff: {max_diff:.2e})"
                        )
        elif AW == 8:  # Mt == 4
            mapping = [(0, 6), (1, 5), (2, 2), (3, 1)]
            for m in range(M // Mt):
                for ref_idx, oact_idx in mapping:
                    actual_ref_idx = m * Mt + ref_idx
                    actual_oact_idx = m * 2 * Mt + oact_idx
                    try:
                        np.testing.assert_allclose(
                            ref[actual_ref_idx], oActs[actual_oact_idx], atol=1e-5
                        )
                        max_diff = np.max(
                            np.abs(ref[actual_ref_idx] - oActs[actual_oact_idx])
                        )
                        print(
                            f"  ✓ ref[{actual_ref_idx}] vs oActs[{actual_oact_idx}]: PASS (max diff: {max_diff:.2e})"
                        )
                    except AssertionError:
                        max_diff = np.max(
                            np.abs(ref[actual_ref_idx] - oActs[actual_oact_idx])
                        )
                        print(
                            f"  ✗ ref[{actual_ref_idx}] vs oActs[{actual_oact_idx}]: FAIL (max diff: {max_diff:.2e})"
                        )
        elif AW == 4:
            mapping = [(0, 2), (1, 0)]
            for m in range(M // Mt):
                for ref_idx, oact_idx in mapping:
                    actual_ref_idx = m * Mt + ref_idx
                    actual_oact_idx = m * 2 * Mt + oact_idx
                    try:
                        np.testing.assert_allclose(
                            ref[actual_ref_idx], oActs[actual_oact_idx], atol=1e-5
                        )
                        max_diff = np.max(
                            np.abs(ref[actual_ref_idx] - oActs[actual_oact_idx])
                        )
                        print(
                            f"  ✓ ref[{actual_ref_idx}] vs oActs[{actual_oact_idx}]: PASS (max diff: {max_diff:.2e})"
                        )
                    except AssertionError:
                        max_diff = np.max(
                            np.abs(ref[actual_ref_idx] - oActs[actual_oact_idx])
                        )
                        print(
                            f"  ✗ ref[{actual_ref_idx}] vs oActs[{actual_oact_idx}]: FAIL (max diff: {max_diff:.2e})"
                        )

    print("-" * 80)

    if hls.is_available("vitis_hls"):
        print("Running Vitis Synthesis and On-Board Execution...")
        s = df.customize(top)
        nest_loop = s.get_loops("NEST_0")["nest"]["i"]
        s.unroll(nest_loop)
        s.partition("top:output_buffer", dim=1, factor=AW)
        csyn_mod = s.build(
            target="vitis_hls",
            mode="hw_emu",
            project=f"feather_gemm_{M}_{N}_{K}_{AW}_{AH}.prj",
        )
        oActs_hls = np.zeros((2 * M, N), dtype=np.int8)
        for n in range(N // Nt):
            for m in range(M // Mt):
                output_buffer = np.zeros((AW, Nt), dtype=np.int8)
                for k in range(K // Kt):
                    weights_tile = np.ascontiguousarray(
                        weights[k * Kt : (k + 1) * Kt, n * Nt : (n + 1) * Nt]
                    )
                    iActs_tile_no_layout = iActs_no_layout[
                        m * Mt : (m + 1) * Mt, k * Kt : (k + 1) * Kt
                    ]
                    iActs_tile = iAct_tile_reorder(iActs_tile_no_layout)
                    csyn_mod(iActs_tile, weights_tile, inst, output_buffer)
                np.copyto(
                    dst=oActs_hls[m * 2 * Mt : (m + 1) * 2 * Mt, n * Nt : (n + 1) * Nt],
                    src=output_buffer,
                )

        print("Verifying HLS Results:")
        hls_test_passed = True

        try:
            correctness_check(oActs_hls, ref)
            print("🎉 Vitis HLS: ALL TESTS PASSED!")

            # Detailed verification for debugging
            if AW == 16:  # Mt == 8
                mapping = [
                    (0, 8),
                    (1, 10),
                    (2, 11),
                    (3, 9),
                    (4, 5),
                    (5, 6),
                    (6, 7),
                    (7, 4),
                ]
                for m in range(M // Mt):
                    for ref_idx, oact_idx in mapping:
                        actual_ref_idx = m * Mt + ref_idx
                        actual_oact_idx = m * 2 * Mt + oact_idx
                        max_diff = np.max(
                            np.abs(ref[actual_ref_idx] - oActs_hls[actual_oact_idx])
                        )
                        print(
                            f"  ✓ ref[{actual_ref_idx}] vs oActs_hls[{actual_oact_idx}]: PASS (max diff: {max_diff:.2e})"
                        )
            elif AW == 8:  # Mt == 4
                mapping = [(0, 6), (1, 5), (2, 2), (3, 1)]
                for m in range(M // Mt):
                    for ref_idx, oact_idx in mapping:
                        actual_ref_idx = m * Mt + ref_idx
                        actual_oact_idx = m * 2 * Mt + oact_idx
                        max_diff = np.max(
                            np.abs(ref[actual_ref_idx] - oActs_hls[actual_oact_idx])
                        )
                        print(
                            f"  ✓ ref[{actual_ref_idx}] vs oActs_hls[{actual_oact_idx}]: PASS (max diff: {max_diff:.2e})"
                        )
            elif AW == 4:
                mapping = [(0, 2), (1, 0)]
                for m in range(M // Mt):
                    for ref_idx, oact_idx in mapping:
                        actual_ref_idx = m * Mt + ref_idx
                        actual_oact_idx = m * 2 * Mt + oact_idx
                        max_diff = np.max(
                            np.abs(ref[actual_ref_idx] - oActs_hls[actual_oact_idx])
                        )
                        print(
                            f"  ✓ ref[{actual_ref_idx}] vs oActs_hls[{actual_oact_idx}]: PASS (max diff: {max_diff:.2e})"
                        )

        except AssertionError as e:
            print("❌ Vitis HLS: SOME TESTS FAILED!")
            hls_test_passed = False

            # Detailed failure analysis for HLS
            if AW == 16:  # Mt == 8
                mapping = [
                    (0, 8),
                    (1, 10),
                    (2, 11),
                    (3, 9),
                    (4, 5),
                    (5, 6),
                    (6, 7),
                    (7, 4),
                ]
                for m in range(M // Mt):
                    for ref_idx, oact_idx in mapping:
                        actual_ref_idx = m * Mt + ref_idx
                        actual_oact_idx = m * 2 * Mt + oact_idx
                        try:
                            np.testing.assert_allclose(
                                ref[actual_ref_idx],
                                oActs_hls[actual_oact_idx],
                                atol=1e-5,
                            )
                            max_diff = np.max(
                                np.abs(ref[actual_ref_idx] - oActs_hls[actual_oact_idx])
                            )
                            print(
                                f"  ✓ ref[{actual_ref_idx}] vs oActs_hls[{actual_oact_idx}]: PASS (max diff: {max_diff:.2e})"
                            )
                        except AssertionError:
                            max_diff = np.max(
                                np.abs(ref[actual_ref_idx] - oActs_hls[actual_oact_idx])
                            )
                            print(
                                f"  ✗ ref[{actual_ref_idx}] vs oActs_hls[{actual_oact_idx}]: FAIL (max diff: {max_diff:.2e})"
                            )
            elif AW == 8:  # Mt == 4
                mapping = [(0, 6), (1, 5), (2, 2), (3, 1)]
                for m in range(M // Mt):
                    for ref_idx, oact_idx in mapping:
                        actual_ref_idx = m * Mt + ref_idx
                        actual_oact_idx = m * 2 * Mt + oact_idx
                        try:
                            np.testing.assert_allclose(
                                ref[actual_ref_idx],
                                oActs_hls[actual_oact_idx],
                                atol=1e-5,
                            )
                            max_diff = np.max(
                                np.abs(ref[actual_ref_idx] - oActs_hls[actual_oact_idx])
                            )
                            print(
                                f"  ✓ ref[{actual_ref_idx}] vs oActs_hls[{actual_oact_idx}]: PASS (max diff: {max_diff:.2e})"
                            )
                        except AssertionError:
                            max_diff = np.max(
                                np.abs(ref[actual_ref_idx] - oActs_hls[actual_oact_idx])
                            )
                            print(
                                f"  ✗ ref[{actual_ref_idx}] vs oActs_hls[{actual_oact_idx}]: FAIL (max diff: {max_diff:.2e})"
                            )
            elif AW == 4:
                mapping = [(0, 2), (1, 0)]
                for m in range(M // Mt):
                    for ref_idx, oact_idx in mapping:
                        actual_ref_idx = m * Mt + ref_idx
                        actual_oact_idx = m * 2 * Mt + oact_idx
                        try:
                            np.testing.assert_allclose(
                                ref[actual_ref_idx],
                                oActs_hls[actual_oact_idx],
                                atol=1e-5,
                            )
                            max_diff = np.max(
                                np.abs(ref[actual_ref_idx] - oActs_hls[actual_oact_idx])
                            )
                            print(
                                f"  ✓ ref[{actual_ref_idx}] vs oActs_hls[{actual_oact_idx}]: PASS (max diff: {max_diff:.2e})"
                            )
                        except AssertionError:
                            max_diff = np.max(
                                np.abs(ref[actual_ref_idx] - oActs_hls[actual_oact_idx])
                            )
                            print(
                                f"  ✗ ref[{actual_ref_idx}] vs oActs_hls[{actual_oact_idx}]: FAIL (max diff: {max_diff:.2e})"
                            )

        # Compare simulator vs HLS results
        print("-" * 80)
        print("Comparing Simulator vs HLS Results:")
        sim_vs_hls_passed = True
        max_sim_hls_diff = np.max(np.abs(oActs - oActs_hls))
        mean_sim_hls_diff = np.mean(np.abs(oActs - oActs_hls))

        try:
            np.testing.assert_allclose(oActs, oActs_hls, atol=1e-5)
            print(f"  ✓ Simulator vs HLS: PASS")
            print(f"    Max difference: {max_sim_hls_diff:.2e}")
            print(f"    Mean difference: {mean_sim_hls_diff:.2e}")
        except AssertionError:
            print(f"  ✗ Simulator vs HLS: FAIL")
            print(f"    Max difference: {max_sim_hls_diff:.2e}")
            print(f"    Mean difference: {mean_sim_hls_diff:.2e}")
            sim_vs_hls_passed = False

    print("=" * 80)
    print("FINAL SUMMARY:")
    if test_passed:
        print("✅ Dataflow Simulator: PASSED")
    else:
        print("❌ Dataflow Simulator: FAILED")

    if hls.is_available("vitis_hls"):
        if hls_test_passed:
            print("✅ Vitis HLS: PASSED")
        else:
            print("❌ Vitis HLS: FAILED")

        if sim_vs_hls_passed:
            print("✅ Simulator vs HLS Consistency: PASSED")
        else:
            print("❌ Simulator vs HLS Consistency: FAILED")
    else:
        print("⚠️  Vitis HLS: NOT AVAILABLE")
    print("=" * 80)


if __name__ == "__main__":
    test_FEATHER_GEMM()
