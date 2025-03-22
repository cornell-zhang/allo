from math import log2
from enum import Enum

import allo
from allo.ir.types import float32, uint2
import allo.dataflow as df
import allo.backend.hls as hls
import numpy as np

AH = 8  # PE array height
AW = 8  # PE array width
P0 = 2 * int(log2(AW))  # Number of stages
P1 = AW // 2  # Number of switches in a stage

PS = 0 # Pass
AR = 1 # Add Right
AL = 2 # Add Left
SW = 3 # Swap

# The 2nd GEMM example in Fig. 10 
# Workload A (change oAct Layout), partially scalable according to AH
M, K, N = 8, AH * 2, AH
# For buffers, a bank is a row
# So it looks like that the iActs and oActs matrices are transposed
ID = M * K // AW # Bank depth of iActs matrix in memory
OD = M * N // P1 # Bank depth of oActs matrix in memory, = AH * ID

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
        df.pipe(dtype=float32, shape=(), depth=1), shape=(AW,)
    ) # To BIRRD
    to_next_pe = df.array(
        df.pipe(dtype=float32, shape=(), depth=1), shape=(AH, AW)
    )
    
    @df.kernel(mapping=[AH+1, AW])
    def NEST(iActs: float32[AW, ID], weights: float32[K, N]):
        i, j = df.get_pid()
        with allo.meta_if(i == 0):
            # The additional row to fetch iActs data
            for k in range(ID):
                to_next_pe[0, j].put(iActs[j, k])
        with allo.meta_else():
            for _ in range(M // (AW // 2)): # Equals IH // AH times of local reductions
                local_reduction_result: float32 = 0.
                for k in range(AH):
                    iAct: float32 = to_next_pe[i - 1, j].get()
                    with allo.meta_if(j < AW // 2):
                        local_reduction_result += iAct * weights[AW // 2 + k, i - 1]
                    with allo.meta_else():
                        local_reduction_result += iAct * weights[k, i - 1]
                    if k == AH - 1:
                        # Finished a local reduction
                        bus[j].put(local_reduction_result)
                    with allo.meta_if(i != AH):
                        to_next_pe[i, j].put(iAct)
    
    connection = df.array(
        df.pipe(dtype=float32, shape=(), depth=1), shape=(P0 - 1, P1 * 2)
    )

    @df.kernel(mapping=[P0, P1])
    def BIRRD(inst: uint2[P0, P1], oActs: float32[AW, OD]):
        i, j = df.get_pid()
        for k in range(OD):
            # The first stage
            with allo.meta_if(i == 0):
                if inst[i, j] == 0:  # Pass
                    connection[i, reverse_bits(2 * j, min(P0 // 2, 2 + i, P0 - i))].put(
                        bus[2 * j].get()
                    )
                    connection[i, reverse_bits(2 * j + 1, min(P0 // 2, 2 + i, P0 - i))].put(
                        bus[2 * j + 1].get()
                    )
                elif inst[i, j] == 1:  # Add Right
                    connection[i, reverse_bits(2 * j, min(P0 // 2, 2 + i, P0 - i))].put(
                        bus[2 * j].get()
                    )
                    connection[i, reverse_bits(2 * j + 1, min(P0 // 2, 2 + i, P0 - i))].put(
                        bus[2 * j].get() + bus[2 * j + 1].get()
                    )
                elif inst[i, j] == 2:  # Add Left
                    connection[i, reverse_bits(2 * j, min(P0 // 2, 2 + i, P0 - i))].put(
                        bus[2 * j].get() + bus[2 * j + 1].get()
                    )
                    connection[i, reverse_bits(2 * j + 1, min(P0 // 2, 2 + i, P0 - i))].put(
                        bus[2 * j + 1].get()
                    )
                else:  # Swap
                    connection[i, reverse_bits(2 * j, min(P0 // 2, 2 + i, P0 - i))].put(
                        bus[2 * j + 1].get()
                    )
                    connection[i, reverse_bits(2 * j + 1, min(P0 // 2, 2 + i, P0 - i))].put(
                        bus[2 * j].get()
                    )

            # The last stage
            with allo.meta_elif(i == P0 - 1):
                in_left: float32 = connection[
                    i - 1, reverse_bits(2 * j, min(P0 // 2, 2 + i, P0 - i))
                ].get()
                in_right: float32 = connection[
                    i - 1, reverse_bits(2 * j + 1, min(P0 // 2, 2 + i, P0 - i))
                ].get()
                if inst[i, j] == 0:  # Pass
                    oActs[k, 2 * j] = in_left
                    oActs[k, 2 * j + 1] = in_right
                elif inst[i, j] == 1:  # Add Right
                    oActs[k, 2 * j] = in_left
                    oActs[k, 2 * j + 1] = in_left + in_right
                elif inst[i, j] == 2:  # Add Left
                    oActs[k, 2 * j] = in_left + in_right
                    oActs[k, 2 * j + 1] = in_right
                else:  # Swap
                    oActs[k, 2 * j] = in_right
                    oActs[k, 2 * j + 1] = in_left

            # Stages in the middle
            with allo.meta_else():
                in_left: float32 = connection[
                    i - 1, reverse_bits(2 * j, min(P0 // 2, 2 + i, P0 - i))
                ].get()
                in_right: float32 = connection[
                    i - 1, reverse_bits(2 * j + 1, min(P0 // 2, 2 + i, P0 - i))
                ].get()
                if inst[i, j] == 0:  # Pass
                    connection[i, reverse_bits(2 * j, min(P0 // 2, 2 + i, P0 - i))].put(
                        in_left
                    )
                    connection[i, reverse_bits(2 * j + 1, min(P0 // 2, 2 + i, P0 - i))].put(
                        in_right
                    )
                elif inst[i, j] == 1:  # Add Right
                    connection[i, reverse_bits(2 * j, min(P0 // 2, 2 + i, P0 - i))].put(
                        in_left
                    )
                    connection[i, reverse_bits(2 * j + 1, min(P0 // 2, 2 + i, P0 - i))].put(
                        in_left + in_right
                    )
                elif inst[i, j] == 2:  # Add Left
                    connection[i, reverse_bits(2 * j, min(P0 // 2, 2 + i, P0 - i))].put(
                        in_left + in_right
                    )
                    connection[i, reverse_bits(2 * j + 1, min(P0 // 2, 2 + i, P0 - i))].put(
                        in_right
                    )
                else:  # Swap
                    connection[i, reverse_bits(2 * j, min(P0 // 2, 2 + i, P0 - i))].put(
                        in_right
                    )
                    connection[i, reverse_bits(2 * j + 1, min(P0 // 2, 2 + i, P0 - i))].put(
                        in_left
                    )


def iAct_make_layout(iActs: np.ndarray):
    B = iActs.reshape(M, 2, AH)
    B = np.swapaxes(B, 0, 1)
    C1 = B[0].reshape(AW // 2, M // (AW // 2), AH)
    C2 = B[1].reshape(AW // 2, M // (AW // 2), AH)
    D = np.vstack((C1, C2))
    # print(iActs, D)
    return np.ascontiguousarray(D)


def oAct_make_layout(oActs: np.ndarray):
    pass


def test_FEATHER_GEMM():
    # mod_orig = df.customize(top)
    # print(mod_orig.module)
    
    inst = np.array([
        PS, PS, PS, PS,
        PS, PS, PS, PS,
        AR, AR, AL, AL,
        SW, SW, SW, SW,
        SW, PS, PS, SW,
        PS, PS, PS, PS
    ])
    iActs_no_layout = np.random.rand(M, K).astype(np.float32)
    iActs = iAct_make_layout(iActs_no_layout)
    
    weights = np.random.rand(K, N).astype(np.float32)
    # oActs_no_layout = np.zeros((M, N), dtype=np.float32)
    oActs = np.zeros((AW, OD), dtype=np.float32)
    
    sim_mod = df.build(top, target="simulator")
    # print(sim_mod.module)
    sim_mod(iActs, weights, inst, oActs)
    print(oActs)
    # np.testing.assert_allclose(, np.dot(A, B), atol=1e-5)

if __name__ == "__main__":
    test_FEATHER_GEMM()
