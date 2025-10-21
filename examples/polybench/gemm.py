# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import json
import pytest
import allo
import numpy as np
from allo.ir.types import int32, float32
import allo.ir.types as T


def gemm_np(A, B, C, beta):
    out_AB = np.dot(A, B)
    out_ABC = beta * C + out_AB
    return out_ABC


def mm1[
    T: (float32, int32), P: int32, Q: int32, R: int32
](A: "T[P, Q]", B: "T[Q, R]", out_AB: "T[P, R]"):
    for i0, j0 in allo.grid(P, R, name="mm1"):
        for k0 in allo.reduction(Q):
            out_AB[i0, j0] += A[i0, k0] * B[k0, j0]


def ele_add[
    T: (float32, int32), P: int32, R: int32
](out_AB: "T[P, R]", C: "T[P, R]", output: "T[P, R]"):
    for i2, j2 in allo.grid(P, R):
        output[i2, j2] = beta * C[i2, j2] + out_AB[i2, j2]


def kernel_gemm[
    T: (float32, int32), P: int32, Q: int32, R: int32
](A: "T[P, Q]", B: "T[Q, R]", C: "T[P, R]", output: "T[P, R]"):
    out_AB: T[P, R] = 0
    mm1[T, P, Q, R](A, B, out_AB)
    ele_add[T, P, R](out_AB, C, output)


def gemm(concrete_type, p, r, q, beta=0.1):
    sch0 = allo.customize(mm1, instantiate=[concrete_type, p, q, r])
    sch0.reorder("k0", "j0")
    sch0.buffer_at(sch0.out_AB, axis="i0")
    sch0.pipeline("k0")
    i0 = sch0.get_loops("mm1")["mm1"]["i0"]

    sch1 = allo.customize(ele_add, instantiate=[concrete_type, p, r])
    sch1.pipeline("j2")

    sch = allo.customize(kernel_gemm, instantiate=[concrete_type, p, q, r])
    sch.compose(sch0)
    sch.compose(sch1)

    return sch


def test_gemm():
    # read problem size settings
    setting_path = os.path.join(os.path.dirname(__file__), "psize.json")
    with open(setting_path, "r") as f:
        psize = json.load(f)
    # for CI test we use small problem size
    test_psize = "small"
    P = psize["gemm"][test_psize]["P"]
    R = psize["gemm"][test_psize]["R"]
    Q = psize["gemm"][test_psize]["Q"]
    beta = 0.1
    concrete_type = float32
    sch = gemm(concrete_type, P, R, Q, beta=beta)
    mod = sch.build()
    # functional correctness test
    A = np.random.rand(P, Q).astype(np.float32)
    B = np.random.rand(Q, R).astype(np.float32)
    C = np.random.rand(P, R).astype(np.float32)
    output = np.zeros((P, R)).astype(np.float32)
    output_ref = gemm_np(A, B, C, beta)
    mod = sch.build()
    mod(A, B, C, output)
    np.testing.assert_allclose(output, output_ref, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__])
