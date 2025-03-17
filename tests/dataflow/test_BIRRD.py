from math import log2

import allo
from allo.ir.types import float32, uint2
import allo.dataflow as df
import allo.backend.hls as hls
import numpy as np

AW = 8
P0 = 2 * int(log2(AW))  # Number of stages
P1 = AW // 2  # Number of switches in a stage


def reverse_bits(data: int, bit_range: int):
    mask = (1 << bit_range) - 1
    reversed_bits: int = 0
    for i in range(0, bit_range):
        if data & (1 << i):
            reversed_bits |= 1 << (bit_range - 1 - i)
    return (data & ~mask) | reversed_bits


@df.region()
def top():
    connection = df.array(
        df.pipe(dtype=float32, shape=(), depth=1), shape=(P0 - 1, P1 * 2)
    )

    @df.kernel(mapping=[P0, P1])
    def BIRRD(inst: uint2[P0, P1], A: float32[AW], B: float32[AW]):
        i, j = df.get_pid()

        # The first stage
        with allo.meta_if(i == 0):
            if inst[i, j] == 0:  # Pass
                connection[i, reverse_bits(2 * j, min(P0 // 2, 2 + i, P0 - i))].put(
                    A[2 * j]
                )
                connection[i, reverse_bits(2 * j + 1, min(P0 // 2, 2 + i, P0 - i))].put(
                    A[2 * j + 1]
                )
            elif inst[i, j] == 1:  # Add Right
                connection[i, reverse_bits(2 * j, min(P0 // 2, 2 + i, P0 - i))].put(
                    A[2 * j]
                )
                connection[i, reverse_bits(2 * j + 1, min(P0 // 2, 2 + i, P0 - i))].put(
                    A[2 * j] + A[2 * j + 1]
                )
            elif inst[i, j] == 2:  # Add Left
                connection[i, reverse_bits(2 * j, min(P0 // 2, 2 + i, P0 - i))].put(
                    A[2 * j] + A[2 * j + 1]
                )
                connection[i, reverse_bits(2 * j + 1, min(P0 // 2, 2 + i, P0 - i))].put(
                    A[2 * j + 1]
                )
            else:  # Swap
                connection[i, reverse_bits(2 * j, min(P0 // 2, 2 + i, P0 - i))].put(
                    A[2 * j + 1]
                )
                connection[i, reverse_bits(2 * j + 1, min(P0 // 2, 2 + i, P0 - i))].put(
                    A[2 * j]
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
                B[2 * j] = in_left
                B[2 * j + 1] = in_right
            elif inst[i, j] == 1:  # Add Right
                B[2 * j] = in_left
                B[2 * j + 1] = in_left + in_right
            elif inst[i, j] == 2:  # Add Left
                B[2 * j] = in_left + in_right
                B[2 * j + 1] = in_right
            else:  # Swap
                B[2 * j] = in_right
                B[2 * j + 1] = in_left

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


def test_BIRRD():
    mod_orig = df.customize(top)
    print(mod_orig.module)


if __name__ == "__main__":
    test_BIRRD()
