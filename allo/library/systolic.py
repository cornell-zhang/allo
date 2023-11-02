# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=used-before-assignment

import allo
from allo.ir.types import int32, index, TypeVar

# The actual matrix size: (M x K) x (K x N) = (M x N)
M, N, K = TypeVar(int32), TypeVar(int32), TypeVar(int32)
# Tiled systolic array size: Mt x Nt
Mt, Nt = TypeVar(int32), TypeVar(int32)
# Data type of the matrices
T_A, T_B, T_C = TypeVar(), TypeVar(), TypeVar()


def PE_kernel(
    A_in: T_A[K],
    B_in: T_B[K],
    A_out: T_A[K],
    B_out: T_B[K],
    C: T_C[Mt, Nt],
    i: index,
    j: index,
):
    v: T_C = 0
    for k in range(K):
        a: T_A = A_in[k]
        b: T_B = B_in[k]
        v += a * b
        A_out[k] = a
        B_out[k] = b
    C[i, j] = v


def systolic_tile(A: T_A[Mt, K], B: T_B[K, Nt], C: T_C[Mt, Nt]):
    A_fifo: T_A[Mt, Nt + 1, K]
    B_fifo: T_B[Nt, Mt + 1, K]
    A_drain: T_A[Mt]
    B_drain: T_B[Nt]

    for k in range(K, name="data_load"):
        # Can be fully unrolled inside this loop,
        # once A and B are correctly partitioned
        for m in range(Mt):
            A_fifo[m, 0, k] = A[m, k]
        for n in range(Nt):
            B_fifo[n, 0, k] = B[k, n]
    for i, j in allo.grid(Mt, Nt, name="PE"):
        PE_kernel(
            A_fifo[i, j], B_fifo[j, i], A_fifo[i, j + 1], B_fifo[j, i + 1], C, i, j
        )
    for k in range(K, name="data_drain"):
        for m in range(Mt):
            A_drain[m] = A_fifo[m, Nt, k]
        for n in range(Nt):
            B_drain[n] = B_fifo[n, Mt, k]


def systolic(A: T_A[M, K], B: T_B[K, N], C: T_C[M, N]):
    local_A: T_A[Mt, K]
    local_B: T_B[K, Nt]
    local_C: T_C[Mt, Nt]

    # k needs not be tiled, since it is temporal dimension
    for mi, ni in allo.grid(M // Mt, N // Nt, name="outer_tile"):
        # reversed traversal, better for cascading systolic arrays with FIFOs
        # corresponds to the order of the previous `store_C_tile` output
        for ak, ai in allo.grid(K, Mt, name="load_A_tile"):
            # reuse along the ni dimension
            if ni == 0:
                local_A[ai, ak] = A[mi * Mt + ai, ak]
        for bk, bj in allo.grid(K, Nt, name="load_B_tile"):
            # reuse along the mi dimension
            # since the inner access order is different from the outer one,
            # we cannot cache as a line buffer
            local_B[bk, bj] = B[bk, ni * Nt + bj]
        systolic_tile(
            local_A,
            local_B,
            local_C,
        )
        # reversed traversal, better for cascading systolic arrays with FIFOs
        for sj, si in allo.grid(Nt, Mt, name="store_C_tile"):
            C[mi * Mt + si, ni * Nt + sj] = local_C[si, sj]
