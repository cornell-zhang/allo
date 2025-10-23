# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# FEATHER is an accelerator proposed in paper:
# Jianming Tong, Anirudh Itagi, Prasanth Chatarasi, Tushar Krishna,
# "FEATHER: A Reconfigurable Accelerator with Data Reordering
# Support for Low-Cost On-Chip Dataflow Switching", 2024 ACM/IEEE 51st
# Annual International Symposium on Computer Architecture (ISCA).

# Paper link: https://arxiv.org/abs/2405.13170
# Original RTL code: https://github.com/maeri-project/FEATHER

# This implementation doesn't contain the quantization module
# presented in the original paper.

from math import log2

import allo
from allo.ir.types import int8, UInt, AlloType, int32, Stream
import allo.dataflow as df
import allo.backend.hls as hls
import numpy as np


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


def get_feather_top(AW: int, AH: int, Ty: AlloType):
    P_FACTOR = 16
    TyPacked = UInt(Ty.bits * P_FACTOR)

    # BIRRD params
    LOG2_AW = int(log2(AW))
    P0 = 2 * LOG2_AW if AW > 4 else 2 * LOG2_AW - 1  # Number of stages
    P1 = AW // 2  # Number of switches in a stage

    @df.region()
    def top():
        nest_out: Stream[TyPacked, AH]

        @df.kernel(mapping=[1])
        def NEST(iActs: Ty[AH, AW], weights: Ty[AH, AW, AH]):
            for i in allo.grid(AH, name="nest"):  # Rows, can be pipelined
                local_result: TyPacked = 0
                for j in range(AW):  # Cols, can be fully parallelized
                    # FIXME: This multiplication op requires a DSP, leading to II=2
                    start: int32 = j * Ty.bits
                    end: int32 = start + Ty.bits
                    for k in range(AH):  # Iterations of a local reduction
                        iAct: Ty = iActs[k, j]
                        weight: Ty = weights[i, j, k]
                        local_result[start:end] += iAct * weight
                nest_out.put(local_result)

        connection: Stream[Ty, 1][P0 + 1, P1 * 2]

        @df.kernel(mapping=[1])
        def bus():
            for _ in range(AH):
                array: TyPacked = nest_out.get()
                with allo.meta_for(AW) as i:
                    connection[0, i].put(array[i * Ty.bits : (i + 1) * Ty.bits])

        inst_input: Stream[int8, 1][P0, P1]

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
                        i + 1,
                        reverse_bits(2 * j + 1, min(LOG2_AW, 2 + i, 2 * LOG2_AW - i)),
                    ].put(out_right)

        @df.kernel(mapping=[1])
        def output(output_buffer: Ty[AH, AW]):
            for d in range(AH):
                with allo.meta_for(AW) as i:
                    # TODO: Do reduction on-chip
                    output_buffer[d, i] = connection[P0, i].get()

    return top


def print_test_config(
    test_name: str,
    AH: int,
    AW: int,
    matrix_dimensions: dict[str, int],
    tile_dimensions: dict[str, int],
    insts: list[np.ndarray],
    iActs: np.ndarray,
    weights: np.ndarray,
    oActs: np.ndarray,
    tiles: list[int],
):

    print("=" * 80)
    print(f"FEATHER {test_name} Test Configuration")
    print("=" * 80)
    print(f"Array Height (AH): {AH}")
    print(f"Array Width (AW): {AW}")
    mat_dim_strs = [f"{key}: {value}" for key, value in matrix_dimensions.items()]
    print(f"Matrix dimensions - {", ".join(mat_dim_strs)}")
    tile_dim_strs = [f"{key}: {value}" for key, value in tile_dimensions.items()]
    print(f"Tile dimensions - {", ".join(tile_dim_strs)}")
    LOG2_AW = int(log2(AW))
    P0 = 2 * LOG2_AW if AW > 4 else 2 * LOG2_AW - 1
    P1 = AW // 2
    print(f"Number of stages (P0): {P0}")
    print(f"Switches per stage (P1): {P1}")
    print("-" * 80)

    print(f"Instruction array shape: {insts[0].shape}")
    print(f"Instruction array:\n")
    for inst in insts:
        print(inst)
    print("-" * 80)

    print(f"Input activations shape: {iActs.shape}")
    print(f"Weights shape: {weights.shape}")
    print(f"Output activations shape (allocated): {oActs.shape}")
    print("-" * 80)

    print(f"# of tiles: {", ".join([str(i) for i in tiles])}")


def result_check(oActs: np.ndarray, ref: np.ndarray, is_hls: bool = False):
    assert oActs.shape == ref.shape

    print("Verifying Results:")
    test_passed = True

    try:
        np.testing.assert_allclose(ref, oActs, atol=1e-5)
        if is_hls:
            print("🎉 Vitis HLS: ALL TESTS PASSED!")
        else:
            print("🎉 Dataflow Simulator: ALL TESTS PASSED!")

        # Detailed verification for debugging
        for i in range(ref.shape[0]):
            max_diff = np.max(np.abs(ref[i] - oActs[i]))
            print(f"  ✓ ref[{i}] vs oActs[{i}]: PASS (max diff: {max_diff:.2e})")

    except AssertionError as e:
        if is_hls:
            print("❌ Vitis HLS: SOME TESTS FAILED!")
        else:
            print("❌ Dataflow Simulator: SOME TESTS FAILED!")
        test_passed = False

        # Detailed failure analysis
        for i in range(ref.shape[0]):
            try:
                np.testing.assert_allclose(ref[i], oActs[i], atol=1e-5)
                max_diff = np.max(np.abs(ref[i] - oActs[i]))
                print(f"  ✓ ref[{i}] vs oActs[{i}]: PASS (max diff: {max_diff:.2e})")
            except AssertionError:
                max_diff = np.max(np.abs(ref[i] - oActs[i]))
                print(f"  ✗ ref[{i}] vs oActs[{i}]: FAIL (max diff: {max_diff:.2e})")
    print("-" * 80)
    return test_passed


def compare_sim_hls_result(oActs: np.ndarray, oActs_hls: np.ndarray):
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

    return sim_vs_hls_passed


def print_summary(test_passed: bool, hls_test_passed: bool, vs_passed: bool):
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

        if vs_passed:
            print("✅ Simulator vs HLS Consistency: PASSED")
        else:
            print("❌ Simulator vs HLS Consistency: FAILED")
    else:
        print("⚠️  Vitis HLS: NOT AVAILABLE")
    print("=" * 80)
