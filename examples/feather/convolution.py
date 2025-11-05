# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Example from Fig. 11 of the paper:
# Convolution, Channel-Last -> Row Major

import argparse, os

from allo.ir.types import int8
import allo.dataflow as df
import allo.backend.hls as hls
from examples.feather.feather import (
    get_feather_top,
    print_test_config,
    result_check,
    compare_sim_hls_result,
    print_summary,
)
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

PS = 0  # Pass
AR = 1  # Add Right
AL = 2  # Add Left
SW = 3  # Swap


def call_feather_kernel(
    iActs: np.ndarray,
    weights: np.ndarray,
    oActs: np.ndarray,
    kernel,  # Callable Module
    insts: list[np.ndarray],
):
    # Outer loop: for all sliding windows
    for nt in range(0, N, 1):
        for pt in range(0, P, 1):
            for qt in range(0, Q, 1):
                iActs_sliding_window = iActs[nt, pt : pt + R, qt : qt + S, :].reshape(
                    R * S, C
                )
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
                                weights[mt : mt + Mt, ct : ct + Ct, vn : vn + VN_size]
                            )
                            inst = insts[intraline_offset]
                            inst_flattened = inst.flatten()
                            output_buffer_local = np.zeros((AH, AW), dtype=np.int8)
                            kernel(
                                iActs_tile,
                                weights_tile,
                                inst_flattened,
                                output_buffer_local,
                            )
                            output_buffer += output_buffer_local
                    for m in range(mt, mt + Mt):
                        oActs[nt, m * P * Q // AW + line_offset, intraline_offset] = (
                            output_buffer[m - mt, intraline_offset]
                        )


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
        insts = [inst0, inst1, inst2, inst3]
    iActs = np.random.randint(low=-4, high=4, size=(N, C, H, W), dtype=np.int8)
    iActs_channel_last = np.ascontiguousarray(
        iActs.transpose(0, 2, 3, 1)
    )  # NCHW -> NHWC
    weights = np.random.randint(low=-4, high=4, size=(M, C, R, S), dtype=np.int8)
    weights_flattened = weights.reshape(M, C, R * S)
    oActs_row_major = np.zeros((N, M * P * Q // AW, AW), dtype=np.int8)

    print_test_config(
        "Convolution",
        AH,
        AW,
        {"N": N, "C": C, "H": H, "W": W, "M": M, "R": R, "S": S},
        {"Mt": Mt, "Ct": Ct, "Virtual Neuron size": VN_size},
        insts,
        iActs,
        weights,
        oActs_row_major,
        [M // Mt, C // Ct, R * S // VN_size],
    )

    print("Running Dataflow Simulator...")
    os.environ["OMP_NUM_THREADS"] = "256"
    top = get_feather_top(AW, AH, Ty)
    sim_mod = df.build(top, target="simulator")
    call_feather_kernel(
        iActs_channel_last, weights_flattened, oActs_row_major, sim_mod, insts
    )

    ref = convolve_2d_row_major(iActs, weights)
    ref_reshaped = ref.reshape(N * M, P * Q)
    print(f"Reference computation completed.")

    oActs_compare = oActs_row_major.reshape(N * M, P * Q)
    test_passed = result_check(oActs_compare, ref_reshaped)

    hls_test_passed, vs_passed = False, False
    if hls.is_available("vitis_hls"):
        print("Running Vitis Synthesis and On-Board Execution...")
        top = get_feather_top(AW, AH, Ty)
        s = df.customize(top)
        nest_loop = s.get_loops("NEST_0")["nest"]["i"]
        s.unroll(nest_loop)
        s.partition("top:output_buffer", dim=1, factor=AW)
        csyn_mod = s.build(
            target="vitis_hls",
            mode="hw_emu",
            project=f"feather_convolution_{N}_{C}_{H}_{W}_{M}_{R}_{S}_{AW}_{AH}.prj",
        )
        oActs_hls = np.zeros((N, M * P * Q // AW, AW), dtype=np.int8)
        call_feather_kernel(
            iActs_channel_last, weights_flattened, oActs_hls, csyn_mod, insts
        )
        oActs_hls_compare = oActs_hls.reshape(N * M, P * Q)
        hls_test_passed = result_check(oActs_hls_compare, ref_reshaped, True)

        vs_passed = compare_sim_hls_result(oActs_compare, oActs_hls_compare)
    print_summary(test_passed, hls_test_passed, vs_passed)


if __name__ == "__main__":
    test_FEATHER_conv()
