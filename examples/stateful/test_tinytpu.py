# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile

import allo
from allo.ir.types import float32, Stream, uint8, stateful
import allo.dataflow as df
import allo.backend.hls as hls
import numpy as np

import pytest
import allo
from allo.ir.types import int4, int8, int16, int32, int128, index, UInt
from allo.ir.utils import MockBuffer
from allo.utils import get_np_struct_type
import allo.backend.hls as hls

"""
M, M, M = 2, 2, 2
P0, P1 = M + 2, M + 2

OP_STORE = 1
OP_MM = 2
OP_LOAD_A = 3
OP_LOAD_B = 4
"""

INSTR_H2D_A = 0
INSTR_H2D_B = 1
INSTR_D2H = 2
INSTR_MM = 3

MEM_CAP = 32


def test_tpu():
    M, M, M = 2, 2, 2

    def systolic_array(instr: int8, matx_in: int8[M, M], matx_out: int16[M, M]):
        A: stateful(int8[M, M])
        B: stateful(int8[M, M])
        C: stateful(int16[M, M])
        A_fifo: stateful(int8[M, M + 1, M])
        B_fifo: stateful(int8[M, M + 1, M])

        if instr == INSTR_H2D_A:
            for i in range(M):
                for j in range(M):
                    A[i, j] = matx_in[i, j]
        if instr == INSTR_H2D_B:
            for i in range(M):
                for j in range(M):
                    B[i, j] = matx_in[i, j]
        if instr == INSTR_D2H:
            for i in range(M):
                for j in range(M):
                    matx_out[i, j] = C[i, j]

        if instr == INSTR_MM:
            for k in range(M, name="data_load"):
                for m in range(M):
                    A_fifo[m, 0, k] = A[m, k]
                for n in range(M):
                    B_fifo[n, 0, k] = B[k, n]
            for i, j in allo.grid(M, M, name="PE"):
                C[i, j] = 0
                for k in range(M):
                    a: int8 = A_fifo[i, j, k]
                    b: int8 = B_fifo[j, i, k]
                    C[i, j] += a * b
                    A_fifo[i, j + 1, k] = a
                    B_fifo[j, i + 1, k] = b
            A_drain: int8[M]
            B_drain: int8[M]
            for k in range(M, name="data_drain"):
                for m in range(M):
                    A_drain[m] = A_fifo[m, M, k]
                for n in range(M):
                    B_drain[n] = B_fifo[n, M, k]

    s = allo.customize(systolic_array)
    # print(s.module)
    mod = s.build()
    A = np.random.randint(-8, 8, size=(M, M)).astype(np.int8)
    B = np.random.randint(-8, 8, size=(M, M)).astype(np.int8)
    allo_C = np.zeros((M, M), dtype=np.int16)
    zero_C = np.zeros((M, M), dtype=np.int16)
    np_C = A.astype(np.int16) @ B.astype(np.int16)

    # instr = np.array(9, dtype=np.int8)
    mod(INSTR_H2D_A, A, allo_C)
    np.testing.assert_allclose(allo_C, zero_C, atol=1e-3)
    print("H2D A Passed")
    mod(INSTR_H2D_B, B, allo_C)
    np.testing.assert_allclose(allo_C, zero_C, atol=1e-3)
    print("H2D B Passed")
    mod(INSTR_MM, B, allo_C)
    np.testing.assert_allclose(allo_C, zero_C, atol=1e-3)
    print("MM Passed")
    mod(INSTR_D2H, B, allo_C)
    np.testing.assert_allclose(allo_C, np_C, atol=1e-3)
    print("D2H Passed")
    print(A)
    print(B)
    print(np_C)

    B2 = np.random.randint(-8, 8, size=(M, M)).astype(np.int8)
    np_C2 = A.astype(np.int16) @ B2.astype(np.int16)

    print(A)
    print(B2)
    print(np_C2)

    mod(INSTR_H2D_B, B2, allo_C)
    mod(INSTR_MM, B2, allo_C)
    mod(INSTR_D2H, B2, allo_C)
    np.testing.assert_allclose(allo_C, np_C2, atol=1e-3)
    print("D2H Passed")

    # Driver integration WIP.
    mod = s.build(target="vitis_hls", mode="sw_emu", project="tpu.prj")
    """
    allo_C = np.zeros((M, M), dtype=np.int16)
    instr = np.array(99, dtype=np.int8)

    mod(9, A, allo_C)
    np.testing.assert_allclose(allo_C, np_C, atol=1e-3)
    print("SW EMU Passed")
    """


if __name__ == "__main__":
    test_tpu()
