# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import int16
import allo.dataflow as df
import numpy as np
from allo.memory import Layout


def gen_pingpong_gemm_mapping_primitive(Pm, Pn, Pk):
    # chaining to k dimension
    mapping_primitives = []
    bases: list[list[str]] = []
    for i in range(Pm):
        bases.append([])
        for j in range(Pn):
            base = f"gemm_0_{i}_{j}"
            for k in range(1, Pk):
                mapping_primitives.append(("chain", [base, f"gemm_{k}_{i}_{j}"]))
                base += f"-gemm_{k}_{i}_{j}"
            bases[i].append(base)

    if Pn // 4 > 1 or Pm // 4 > 1:
        for i in range(4):
            for j in range(4):
                bundle_list = []
                for p in range(Pm // 4):
                    for q in range(Pn // 4):
                        bundle_list.append(bases[i + 4 * p][j + 4 * q])
                mapping_primitives.append(("bundle", bundle_list))
    return mapping_primitives


def _test_pingpong_gemm(TyI, TyO):

    # [NOTE]: Pm, Pn are safe to scale up. Pk can be scale up to 64.
    M, N, K = 1024, 128, 64
    Pm, Pn, Pk = 128, 4, 4
    Mt, Nt, Kt = M // Pm, N // Pn, K // Pk

    LyA = Layout("S1S2")
    LyB = Layout("S2S0")
    LyC = Layout("S1S0")

    @df.region()
    def top():
        pipe = df.array(
            df.pipe(dtype=TyO, shape=(Mt, Nt), depth=2), shape=(Pk - 1, Pm, Pn)
        )

        @df.kernel(mapping=[Pk, Pm, Pn])
        def gemm(A: TyI[M, K] @ LyA, B: TyI[K, N] @ LyB, C: TyO[M, N] @ LyC):
            pk, pm, pn = df.get_pid()
            with allo.meta_if(pk > 0):
                C_in: TyO[Mt, Nt] = pipe[pk - 1, pm, pn].get()
            with allo.meta_else():
                C_in: TyO[Mt, Nt] = 0
            C_out: TyO[Mt, Nt] = allo.add(allo.matmul(A, B), C_in)
            with allo.meta_if(pk < Pk - 1):
                pipe[pk, pm, pn].put(C_out)
            with allo.meta_elif(pk == Pk - 1):
                C[:, :] = C_out

    mapping_primitives = gen_pingpong_gemm_mapping_primitive(Pm, Pn, Pk)
    mod = df.build(
        top,
        target="aie-mlir",
        mapping_primitives=mapping_primitives,
        profile=True,
        warmup=200,
        num_iters=1000,
    )
    A = np.random.randint(0, 64, (M, K)).astype(np.int16)
    B = np.random.randint(0, 64, (K, N)).astype(np.int16)
    C = np.zeros((M, N)).astype(np.int16)
    mod(A, B, C)
    np.testing.assert_allclose(C, A @ B, atol=1e-5)
    print("PASSED!")


if __name__ == "__main__":
    _test_pingpong_gemm(int16, int16)
