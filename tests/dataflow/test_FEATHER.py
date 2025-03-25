from math import log2

import allo
from allo.ir.types import float32, int8
import allo.dataflow as df
import allo.backend.hls as hls
import numpy as np

AH = 1  # PE array height
AW = 4  # PE array width
LOG2_AW = int(log2(AW))
P0 = 2 * LOG2_AW - 1  # Number of stages
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
    bus = df.array(
        df.pipe(dtype=float32, shape=(), depth=1), shape=(AH, AW)
        # df.pipe(dtype=float32, shape=(), depth=1),
        # shape=(AW,)
    )  # To BIRRD
    to_next_pe = df.array(df.pipe(dtype=float32, shape=(), depth=1), shape=(AH - 1, AW))

    @df.kernel(mapping=[AH, AW])
    def NEST(iActs: float32[AW, ID], weights: float32[K, N]):
        i, j = df.get_pid()
        for h in range(M // (AW // 2)):  # Equals ID // AH times of local reductions
            local_reduction_result: float32 = 0.0
            for k in range(AH):
                with allo.meta_if(i == 0):
                    iAct: float32 = iActs[j, k + h * AH]
                with allo.meta_else():
                    iAct: float32 = to_next_pe[i - 1, j].get()
                with allo.meta_if(j < AW // 2):
                    local_reduction_result += iAct * weights[K // 2 + k, i]
                with allo.meta_else():
                    local_reduction_result += iAct * weights[k, i]
                if k == AH - 1:
                    # Finished a local reduction
                    # Push the result before passing iAct data to the next PE
                    bus[i, j].put(local_reduction_result)
                    # bus[j].put(local_reduction_result)
                with allo.meta_if(i != AH - 1):
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
                in_left: float32 = bus[k % AH, 2 * j].get()
                in_right: float32 = bus[k % AH, 2 * j + 1].get()
                # in_left: float32 = bus[2 * j].get()
                # in_right: float32 = bus[2 * j + 1].get()
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
                connection[i, 2 * j].put(out_left)
                connection[i, 2 * j + 1].put(out_right)

            # The last stage
            with allo.meta_elif(i == P0 - 1):
                in_left: float32 = connection[
                    i - 1, reverse_bits(2 * j, min(LOG2_AW, 2 + i, 2 * LOG2_AW - i))
                ].get()
                in_right: float32 = connection[
                    i - 1, reverse_bits(2 * j + 1, min(LOG2_AW, 2 + i, 2 * LOG2_AW - i))
                ].get()
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
                in_left: float32 = connection[
                    i - 1, reverse_bits(2 * j, min(LOG2_AW, 2 + i, 2 * LOG2_AW - i))
                ].get()
                in_right: float32 = connection[
                    i - 1, reverse_bits(2 * j + 1, min(LOG2_AW, 2 + i, 2 * LOG2_AW - i))
                ].get()
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
                connection[i, 2 * j].put(out_left)
                connection[i, 2 * j + 1].put(out_right)


def iAct_make_layout(iActs: np.ndarray):
    B = iActs.reshape(M, 2, AH)
    B = np.swapaxes(B, 0, 1)
    C1 = B[0].reshape(AW // 2, M // (AW // 2), AH)
    C2 = B[1].reshape(AW // 2, M // (AW // 2), AH)
    D = np.vstack((C1, C2))
    # print(iActs, D)
    return np.ascontiguousarray(D)


def oAct_make_layout(oActs: np.ndarray):
    B = oActs.reshape(P1, OD)
    return np.ascontiguousarray(B)


def test_FEATHER_GEMM():
    # mod_orig = df.customize(top)
    # print(mod_orig.module)

    # inst_6x4 = np.array([
    #     PS, PS, PS, PS,
    #     PS, PS, PS, PS,
    #     AR, AR, AL, AL,
    #     SW, SW, SW, SW,
    #     SW, PS, PS, SW,
    #     PS, PS, PS, PS,
    # ]).reshape(6, 4)
    inst_3x2 = np.array([PS, PS, AR, AL, PS, PS], dtype=np.int8).reshape(3, 2)
    iActs_no_layout = np.random.rand(M, K).astype(np.float32)
    iActs = iAct_make_layout(iActs_no_layout)
    print(iActs_no_layout)
    print(iActs)
    
    weights = np.random.rand(K, N).astype(np.float32)
    # oActs_no_layout = np.zeros((M, N), dtype=np.float32)
    oActs = np.zeros((AW, OD), dtype=np.float32)

    sim_mod = df.build(top, target="simulator")
    sim_mod(iActs, weights, inst_3x2, oActs)
    print(oActs)
    print(oAct_make_layout(np.dot(iActs_no_layout, weights)))
    # np.testing.assert_allclose(, np.dot(A, B), atol=1e-5)

    # if hls.is_available("vitis_hls"):
    #     mod1 = df.build(top)
    #     oActs = np.zeros((AW, OD), dtype=np.float32)
    #     mod1(iActs, weights, inst_3x2, oActs)
    #     print(oActs)


if __name__ == "__main__":
    test_FEATHER_GEMM()
