from math import log2

import allo
from allo.ir.types import float32, int8
import allo.dataflow as df
import allo.backend.hls as hls
import numpy as np

AH = 8  # PE array height
AW = 8  # PE array width, must be a power of 2
LOG2_AW = int(log2(AW))
P0 = 2 * LOG2_AW  # Number of stages (AW > 4)
# P0 = 2 * LOG2_AW - 1 # Number of stages (AW == 4)
P1 = AW // 2  # Number of switches in a stage

PS = 0  # Pass
AR = 1  # Add Right
AL = 2  # Add Left
SW = 3  # Swap

# The 2nd GEMM example in Fig. 10
# Workload A (change oAct Layout), partially scalable according to AH
M, K, N = 8, AH * 2, AH
# For buffers, a bank is a row
# So it looks like that the iActs and oActs matrices are transposed
ID = M * K // AW  # Bank depth of iActs matrix in memory
OD = M * N // P1  # Bank depth of oActs matrix in memory, = AH * ID


def reverse_bits(data: int, bit_range: int):
    mask = (1 << bit_range) - 1
    reversed_bits: int = 0
    for i in range(0, bit_range):
        if data & (1 << i):
            reversed_bits |= 1 << (bit_range - 1 - i)
    return (data & ~mask) | reversed_bits


@df.region()
def top():
    bus = df.array(df.pipe(dtype=float32, shape=(), depth=AH), shape=(AW,))  # To BIRRD
    to_next_pe = df.array(df.pipe(dtype=float32, shape=(), depth=1), shape=(AH + 1, AW))

    @df.kernel(mapping=[AH + 1, AW])
    def NEST(iActs: float32[AW, ID], weights: float32[K, N]):
        i, j = df.get_pid()
        for h in range(M // (AW // 2)):  # Equals ID // AH times of local reductions
            local_reduction_result: float32 = 0.0
            for k in range(AH):
                with allo.meta_if(i == 0):
                    # Fetch from data source
                    to_next_pe[0, j].put(iActs[j, k + h * AH])

                with allo.meta_else():
                    with allo.meta_if(i == 1):
                        # This is used to ensure the ordering of accessing the bus
                        # The nth reductions (n > 1) of eggs in the first row must complete after the finishing of the (n-1)th reductions of eggs in the last row
                        if h != 0:
                            barrier = to_next_pe[AH, j].get()

                    iAct: float32 = to_next_pe[i - 1, j].get()
                    with allo.meta_if(j < AW // 2):
                        local_reduction_result += iAct * weights[K // 2 + k, i - 1]
                    with allo.meta_else():
                        local_reduction_result += iAct * weights[k, i - 1]

                    if k == AH - 1:
                        # Finished a local reduction
                        # Push the result before passing iAct data to the next PE
                        bus[j].put(local_reduction_result)
                    with allo.meta_if(i == AH):
                        # Do not send barrier values after the last reduction
                        if h != M // (AW // 2) - 1:
                            to_next_pe[i, j].put(iAct)
                    with allo.meta_else():
                        to_next_pe[i, j].put(iAct)

    connection = df.array(
        df.pipe(dtype=float32, shape=(), depth=1), shape=(P0 - 1, P1 * 2)
    )

    @df.kernel(mapping=[P0, P1])
    def BIRRD(inst: int8[P0, P1], oActs: float32[AW, OD]):
        i, j = df.get_pid()
        for k in range(OD):
            # The first stage
            with allo.meta_if(i == 0):
                in_left: float32 = bus[2 * j].get()
                in_right: float32 = bus[2 * j + 1].get()
                out_left: float32 = 0.0
                out_right: float32 = 0.0
                if inst[i, j] == 0:  # Pass
                    out_left = in_left
                    out_right = in_right
                elif inst[i, j] == 1:  # Add Right
                    out_left = in_left
                    out_right = in_left + in_right
                elif inst[i, j] == 2:  # Add Left
                    out_left = in_left + in_right
                    out_right = in_right
                else:  # Swap
                    out_left = in_right
                    out_right = in_left
                connection[
                    i, reverse_bits(2 * j, min(LOG2_AW, 2 + i, 2 * LOG2_AW - i))
                ].put(out_left)
                connection[
                    i, reverse_bits(2 * j + 1, min(LOG2_AW, 2 + i, 2 * LOG2_AW - i))
                ].put(out_right)

            # The last stage
            with allo.meta_elif(i == P0 - 1):
                in_left: float32 = connection[i - 1, 2 * j].get()
                in_right: float32 = connection[i - 1, 2 * j + 1].get()
                if inst[i, j] == 0:  # Pass
                    oActs[2 * j, k] = in_left
                    oActs[2 * j + 1, k] = in_right
                elif inst[i, j] == 1:  # Add Right
                    oActs[2 * j, k] = in_left
                    oActs[2 * j + 1, k] = in_left + in_right
                elif inst[i, j] == 2:  # Add Left
                    oActs[2 * j, k] = in_left + in_right
                    oActs[2 * j + 1, k] = in_right
                else:  # Swap
                    oActs[2 * j, k] = in_right
                    oActs[2 * j + 1, k] = in_left

            # Stages in the middle
            with allo.meta_else():
                in_left: float32 = connection[i - 1, 2 * j].get()
                in_right: float32 = connection[i - 1, 2 * j + 1].get()
                out_left: float32 = 0.0
                out_right: float32 = 0.0
                if inst[i, j] == 0:  # Pass
                    out_left = in_left
                    out_right = in_right
                elif inst[i, j] == 1:  # Add Right
                    out_left = in_left
                    out_right = in_left + in_right
                elif inst[i, j] == 2:  # Add Left
                    out_left = in_left + in_right
                    out_right = in_right
                else:  # Swap
                    out_left = in_right
                    out_right = in_left
                connection[
                    i, reverse_bits(2 * j, min(LOG2_AW, 2 + i, 2 * LOG2_AW - i))
                ].put(out_left)
                connection[
                    i, reverse_bits(2 * j + 1, min(LOG2_AW, 2 + i, 2 * LOG2_AW - i))
                ].put(out_right)


def iAct_make_layout(iActs: np.ndarray):
    # Split the array vertically into 2 parts
    B_left, B_right = np.hsplit(iActs, 2)
    # Allocate rows from the original array across PEs in both side (swapped)
    C_left = np.vsplit(B_right, P1 // 2)
    C_right = np.vsplit(B_left, P1 // 2)
    # Merge all data for the same PE column into a new row
    # And combine the rows into the reordered array
    D = np.vstack([C.flatten() for C in C_left + C_right])
    return np.ascontiguousarray(D)


def oAct_make_layout(oActs_raw: np.ndarray):
    B = oActs_raw.flatten().reshape(P1, -1)
    return np.ascontiguousarray(B)


def test_FEATHER_GEMM():
    inst_6x4 = np.array(
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
    # inst_3x2 = np.array([[PS, PS], [AR, AL], [SW, PS]], dtype=np.int8)

    iActs_no_layout = np.random.rand(M, K).astype(np.float32)
    iActs = iAct_make_layout(iActs_no_layout)
    weights = np.random.rand(K, N).astype(np.float32)

    sim_mod = df.build(top, target="simulator")
    oActs = np.zeros((AW, OD), dtype=np.float32)
    sim_mod(iActs, weights, inst_6x4, oActs)
    # sim_mod(iActs, weights, inst_3x2, oActs)
    oActs_no_layout = np.dot(iActs_no_layout, weights)
    ref = oAct_make_layout(oActs_no_layout)

    # print(ref[3], oActs[1])
    # print(ref[2], oActs[2])
    # print(ref[1], oActs[5])
    # print(ref[0], oActs[6])
    np.testing.assert_allclose(ref[0], oActs[6], atol=1e-5)
    np.testing.assert_allclose(ref[1], oActs[5], atol=1e-5)
    np.testing.assert_allclose(ref[2], oActs[2], atol=1e-5)
    np.testing.assert_allclose(ref[3], oActs[1], atol=1e-5)
    # print(ref[0], oActs[2])
    # print(ref[1], oActs[0])
    # np.testing.assert_allclose(ref[0], oActs[2], atol=1e-5)
    # np.testing.assert_allclose(ref[1], oActs[0], atol=1e-5)
    print("Dataflow Simulator Passed!")

    # if hls.is_available("vitis_hls"):
    #     mod1 = df.build(top)
    #     oActs = np.zeros((AW, OD), dtype=np.float32)
    #     mod1(iActs, weights, inst_3x2, oActs)
    #     oActs_no_layout = np.dot(iActs_no_layout, weights)
    #     ref = oAct_make_layout(oActs_no_layout)
    #     np.testing.assert_allclose(ref[0], oActs[2], atol=1e-5)
    #     np.testing.assert_allclose(ref[1], oActs[0], atol=1e-5)
    #     print("Passed!")


if __name__ == "__main__":
    test_FEATHER_GEMM()
