# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=used-before-assignment

import allo
from allo.ir.types import int32, index, TypeVar

M, N, K = TypeVar(int32), TypeVar(int32), TypeVar(int32)
T_A, T_B, T_C = TypeVar(), TypeVar(), TypeVar()


def sa_kernel(
    A_in: T_A[K],
    B_in: T_B[K],
    A_out: T_A[K],
    B_out: T_B[K],
    C: T_C[M, N],
    i: index,
    j: index,
):
    for k in range(K):
        a: T_A = A_in[k]
        b: T_B = B_in[k]
        C[i, j] += a * b
        A_out[k] = a
        B_out[k] = b


def systolic(A: T_A[M, K], B: T_B[K, N], C: T_C[M, N]):
    A_fifo: T_A[M, N + 1, K]
    B_fifo: T_B[N, M + 1, K]
    A_drain: T_A[M]
    B_drain: T_B[N]

    for k in range(K, name="data_load"):
        for m in range(M):
            A_fifo[m, 0, k] = A[m, k]
        for n in range(N):
            B_fifo[n, 0, k] = B[k, n]
    for i, j in allo.grid(M, N, name="PE"):
        sa_kernel(
            A_fifo[i, j], B_fifo[j, i], A_fifo[i, j + 1], B_fifo[j, i + 1], C, i, j
        )
    for k in range(K, name="data_drain"):
        for m in range(M):
            A_drain[m] = A_fifo[m, N, k]
        for n in range(N):
            B_drain[n] = B_fifo[n, M, k]
