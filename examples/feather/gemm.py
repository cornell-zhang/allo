# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# 2nd example from Fig.10 of the paper
# "Workload A, Change oAct Layout"

from math import log2
import argparse
import os

from allo.ir.types import int8, UInt
import allo.dataflow as df
import allo.backend.hls as hls
from examples.feather.feather import (
    get_feather_top,
    get_scheduled_feather,
    print_test_config,
    result_check,
    compare_sim_hls_result,
    print_summary,
)

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
Mt, Nt, Kt = AW // 2, AH, 2 * AH
M, N, K = args.M, args.N, args.K
assert K % Kt == 0
assert N % Nt == 0
assert M % Mt == 0
assert Kt % 2 == 0
assert K >= Kt
assert N >= Nt
assert M >= Mt


def iAct_tile_reorder(tile: np.ndarray):
    assert tile.shape == (Mt, Kt)
    # Split on K dimension for later reduction
    B_left, B_right = np.hsplit(tile, 2)  # AW // 2, AH
    C = np.hstack([B_left.transpose(), B_right.transpose()])
    assert C.shape == (AH, AW)
    return np.ascontiguousarray(C)


def weight_make_layout_for_input(tile: np.ndarray):
    assert tile.shape == (Kt, Nt)  # (2 * AH, AH)
    B_left, B_right = np.vsplit(tile, 2)
    C_left = np.array([B_left.transpose()] * (AW // 2))
    C_right = np.array([B_right.transpose()] * (AW // 2))
    D = np.vstack([C_left, C_right]).transpose(1, 0, 2)
    assert D.shape == (AH, AW, AH)
    return np.ascontiguousarray(D)


def call_feather_kernel(
    iActs: np.ndarray, weights: np.ndarray, oActs: np.ndarray, kernel, inst: np.ndarray
):
    for n in range(N // Nt):
        for m in range(M // Mt):
            output_buffer = np.zeros((Nt, 2 * Mt), dtype=np.int8)
            for k in range(K // Kt):
                output_buffer_tile = np.zeros((Nt, 2 * Mt), dtype=np.int8)
                weights_tile = weights[k * Kt : (k + 1) * Kt, n * Nt : (n + 1) * Nt]
                weights_tile_input = weight_make_layout_for_input(weights_tile)
                iActs_tile_no_layout = iActs[
                    m * Mt : (m + 1) * Mt, k * Kt : (k + 1) * Kt
                ]
                iActs_tile = iAct_tile_reorder(iActs_tile_no_layout)
                kernel(iActs_tile, weights_tile_input, inst, output_buffer_tile)
                output_buffer += output_buffer_tile
            np.copyto(
                dst=oActs[n * Nt : (n + 1) * Nt, m * 2 * Mt : (m + 1) * 2 * Mt],
                src=output_buffer,
            )


def extract_output_for_compare(oActs: np.ndarray, ref: np.ndarray) -> np.ndarray:
    oActs = oActs.transpose()
    extracted_data = np.zeros(shape=ref.shape, dtype=np.int8)
    for m in range(M // Mt):
        if AW == 16:  # Mt == 8
            np.copyto(extracted_data[m * Mt], oActs[m * 2 * Mt + 8])
            np.copyto(extracted_data[m * Mt + 1], oActs[m * 2 * Mt + 10])
            np.copyto(extracted_data[m * Mt + 2], oActs[m * 2 * Mt + 11])
            np.copyto(extracted_data[m * Mt + 3], oActs[m * 2 * Mt + 9])
            np.copyto(extracted_data[m * Mt + 4], oActs[m * 2 * Mt + 5])
            np.copyto(extracted_data[m * Mt + 5], oActs[m * 2 * Mt + 6])
            np.copyto(extracted_data[m * Mt + 6], oActs[m * 2 * Mt + 7])
            np.copyto(extracted_data[m * Mt + 7], oActs[m * 2 * Mt + 4])
        elif AW == 8:  # Mt == 4
            np.copyto(extracted_data[m * Mt], oActs[m * 2 * Mt + 6])
            np.copyto(extracted_data[m * Mt + 1], oActs[m * 2 * Mt + 5])
            np.copyto(extracted_data[m * Mt + 2], oActs[m * 2 * Mt + 2])
            np.copyto(extracted_data[m * Mt + 3], oActs[m * 2 * Mt + 1])
        elif AW == 4:  # Mt == 2
            np.copyto(extracted_data[m * Mt], oActs[m * 2 * Mt + 2])
            np.copyto(extracted_data[m * Mt + 1], oActs[m * 2 * Mt])
    return extracted_data


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
    insts = [inst]

    iActs_no_layout = np.random.randint(-4, 4, size=(M, K)).astype(np.int8)
    weights = np.random.randint(-4, 4, size=(K, N)).astype(np.int8)
    oActs = np.zeros((N, 2 * M), dtype=np.int8)

    print_test_config(
        "GEMM",
        AH,
        AW,
        {"M": M, "K": K, "N": N},
        {"Mt": Mt, "Kt": Kt, "Nt": Nt},
        insts,
        iActs_no_layout,
        weights,
        oActs,
        [M // Mt, N // Nt, K // Kt],
    )

    print("Running Dataflow Simulator...")
    os.environ["OMP_NUM_THREADS"] = "256"
    top = get_feather_top(AW, AH, Ty)
    sim_mod = df.build(top, target="simulator")
    call_feather_kernel(iActs_no_layout, weights, oActs, sim_mod, inst)

    ref = np.dot(iActs_no_layout, weights)  # MxN
    oActs_compare = extract_output_for_compare(oActs, ref)
    print(f"Reference computation completed.")

    test_passed = result_check(oActs_compare, ref)

    hls_test_passed, vs_passed = False, False
    if hls.is_available("vitis_hls"):
        print("Running Vitis Synthesis and On-Board Execution...")
        s = get_scheduled_feather(AW, AH, Ty)
        csyn_mod = s.build(
            target="vitis_hls",
            mode="hw_emu",
            project=f"feather_gemm_{M}_{N}_{K}_{AW}_{AH}_new.prj",
        )
        oActs_hls = np.zeros((N, 2 * M), dtype=np.int8)
        call_feather_kernel(iActs_no_layout, weights, oActs_hls, csyn_mod, inst)
        oActs_hls_compare = extract_output_for_compare(oActs_hls, ref)

        hls_test_passed = result_check(oActs_hls_compare, ref, True)

        vs_passed = compare_sim_hls_result(oActs_compare, oActs_hls_compare)

    print_summary(test_passed, hls_test_passed, vs_passed)


if __name__ == "__main__":
    test_FEATHER_GEMM()
