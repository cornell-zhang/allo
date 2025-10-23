# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import int16, int32, float32, int8, Stream
import allo.dataflow as df
import numpy as np
from allo.memory import Layout


def test_cooperative_gemm(Ty):
    M, N, K = 32, 32, 32
    Pm, Pn, Pk = 2, 2, 2
    Mt, Nt = M // Pm, N // Pn

    LyA = Layout("S1S2")
    LyB = Layout("S2S0")
    LyC = Layout("S1S0")

    @df.region()
    def top():
        pipe: Stream[Ty[Mt, Nt], 2][Pk - 1, Pm, Pn]

        @df.kernel(mapping=[Pk, Pm, Pn])
        def gemm(A: Ty[M, K] @ LyA, B: Ty[K, N] @ LyB, C: Ty[M, N] @ LyC):
            pk, pm, pn = df.get_pid()
            C_in: Ty[Mt, Nt]
            with allo.meta_if(pk > 0):
                C_in[:, :] = pipe[pk - 1, pm, pn].get()
            with allo.meta_else():
                C_in[:, :] = 0
            C_out: Ty[Mt, Nt] = allo.add(allo.matmul(A, B), C_in)
            with allo.meta_if(pk < Pk - 1):
                pipe[pk, pm, pn].put(C_out)
            with allo.meta_elif(pk == Pk - 1):
                C[:, :] = C_out

    mod = df.build(top, target="aie")
    if Ty == int8:
        A = np.random.randint(-2, 2, (M, K)).astype(np.int8)
        B = np.random.randint(-2, 2, (K, N)).astype(np.int8)
        C = np.zeros((M, N)).astype(np.int8)
    elif Ty == int16:
        A = np.random.randint(-32, 32, (M, K)).astype(np.int16)
        B = np.random.randint(-32, 32, (K, N)).astype(np.int16)
        C = np.zeros((M, N)).astype(np.int16)
    elif Ty == int32:
        A = np.random.randint(-32, 32, (M, K)).astype(np.int32)
        B = np.random.randint(-32, 32, (K, N)).astype(np.int32)
        C = np.zeros((M, N)).astype(np.int32)
    elif Ty == float32:
        A = np.random.random((M, K)).astype(np.float32)
        B = np.random.random((K, N)).astype(np.float32)
        C = np.zeros((M, N)).astype(np.float32)

    mod(A, B, C)
    np.testing.assert_allclose(C, A @ B, atol=1e-5)
    print("PASSED!")


if __name__ == "__main__":
    test_cooperative_gemm(int8)
    test_cooperative_gemm(int16)
    test_cooperative_gemm(int32)
    test_cooperative_gemm(float32)
