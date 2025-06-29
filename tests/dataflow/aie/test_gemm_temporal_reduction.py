# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import int16, int32, float32
import allo.dataflow as df
import numpy as np
from allo.memory import Layout

LyA = Layout("S0R")
LyB = Layout("RS1")
LyC = Layout("S0S1")


TyI, TyO = float32, float32
total_M, total_N, total_K = 64, 64, 512
M, N, K = 64, 64, 64


@df.region()
def top1():
    @df.kernel(mapping=[4, 4])
    def gemm(A: TyI[M, K] @ LyA, B: TyI[K, N] @ LyB, C: TyO[M, N] @ LyC):
        C[:, :] = allo.matmul(A, B)


@df.region()
def top2():
    @df.kernel(mapping=[2, 4])
    def core(A: TyO[M, N] @ LyC, B: TyO[M, N] @ LyC, C: TyO[M, N] @ LyC):
        C[:, :] = allo.add(A, B)


mod1 = df.build(top1, target="aie-mlir", project="top1.prj")
mod2 = df.build(top2, target="aie-mlir", project="top2.prj")

A = np.random.randint(0, 8, (total_M, total_K)).astype(np.float32)
B = np.random.randint(0, 8, (total_K, total_N)).astype(np.float32)
C_tmp = np.zeros((M, N)).astype(np.float32)
C = np.zeros((M, N)).astype(np.float32)

for k in range(total_K // K):
    tile_A = A[:, k * K : (k + 1) * K]
    tile_B = B[k * K : (k + 1) * K, :]
    mod1(tile_A, tile_B, C_tmp)
    mod2(C, C_tmp, C)

np.testing.assert_allclose(C, A @ B, atol=1e-2)
print("PASSED!")
