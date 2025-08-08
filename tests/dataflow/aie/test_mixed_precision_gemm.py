# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import int16, int32, float32
import allo.dataflow as df
import numpy as np
from allo.memory import Layout

def _test_matrix_matrix_add():
    Ty_A, Ty_B = int16, int32
    M, N = 64, 64
    P0 = 4
    Ly = Layout("S0R")

    @df.region()
    def top():
        @df.kernel(mapping=[P0])
        def core(A: Ty_A[M, N] @ Ly, B: Ty_B[M, N] @ Ly, C: Ty_A[M, N] @ Ly):
            C[:, :] = allo.add(A, B)

    A = np.random.randint(0, 100, (M, N)).astype(np.int16)
    B = np.random.randint(0, 100, (M, N)).astype(np.int32)
    mod = df.build(top, target="aie-mlir")
    C = np.zeros((M, N)).astype(np.int16)
    mod(A, B, C)
    np.testing.assert_allclose(C, A.astype(np.int32) + B)
    print("PASSED!")


def _test_gemm_2D():
    TyI_A, TyI_B, TyO = int16, int32, int32
    M, N, K = 32, 32, 32
    P0, P1 = 4, 4

    LyA = Layout("S0R")
    LyB = Layout("RS1")
    LyC = Layout("S0S1")

    @df.region()
    def top():
        @df.kernel(mapping=[P0, P1])
        def gemm(A: TyI_A[M, K] @ LyA, B: TyI_B[K, N] @ LyB, C: TyO[M, N] @ LyC):
            C[:, :] = allo.matmul(A, B)

    mod = df.build(top, target="aie-mlir")
    A = np.random.randint(0, 64, (M, K)).astype(np.int16)
    B = np.random.randint(0, 64, (K, N)).astype(np.int32)
    C = np.zeros((M, N)).astype(np.int32)
    mod(A, B, C)
    np_C = A.astype(np.int32) @ B.astype(np.int32)
    np.testing.assert_allclose(C, np_C, atol=1e-5)
    print("PASSED!")

def test_cooperative_gemm():
    Ty_A = int16
    Ty_B = int16
    Ty_C = int32
    M, N, K = 16, 16, 16
    Pm, Pn, Pk = 2, 2, 2
    Mt, Nt = M // Pm, N // Pn

    LyA = Layout("S1S2")
    LyB = Layout("S2S0")
    LyC = Layout("S1S0")


    @df.region()
    def top():
        pipe = df.array(df.pipe(dtype=Ty_C, shape=(Mt, Nt), depth=2), shape=(Pk - 1, Pm, Pn))

        @df.kernel(mapping=[Pk, Pm, Pn])
        def gemm(A: Ty_A[M, K] @ LyA, B: Ty_B[K, N] @ LyB, C: Ty_C[M, N] @ LyC):
            pk, pm, pn = df.get_pid()
            with allo.meta_if(pk > 0):
                C_in: Ty_C[Mt, Nt] = pipe[pk - 1, pm, pn].get()
            with allo.meta_else():
                C_in: Ty_C[Mt, Nt] = 0
            C_out: Ty_C[Mt, Nt] = (allo.matmul(A, B) + C_in)
            with allo.meta_if(pk < Pk - 1):
                pipe[pk, pm, pn].put(C_out)
            with allo.meta_elif(pk == Pk - 1):
                C[:, :] = C_out

    mod = df.build(top, target="aie-mlir")
    A = np.random.randint(0, 64, (M, K)).astype(np.int16)
    B = np.random.randint(0, 64, (K, N)).astype(np.int32)
    C = np.zeros((M, N)).astype(np.int32)
    mod(A, B, C)
    np.testing.assert_allclose(C, A @ B, atol=1e-5)
    print("PASSED!")


if __name__ == "__main__":
    # _test_matrix_matrix_add()
    # _test_gemm_2D()
    test_cooperative_gemm()
