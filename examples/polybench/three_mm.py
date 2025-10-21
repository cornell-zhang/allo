# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import json
import pytest
import allo
import numpy as np
from allo.ir.types import int32, float32
import allo.ir.types as T


def three_mm_np(A, B, C, D):
    out_AB = np.dot(A, B)
    out_CD = np.dot(C, D)
    out_ABC = np.dot(out_AB, out_CD)
    return out_ABC


def mm1[
    DType: (float32, int32), P: int32, Q: int32, R: int32
](A: "DType[P, Q]", B: "DType[Q, R]", out_AB: "DType[P, R]"):
    for i0, j0 in allo.grid(P, R, name="mm1"):
        for k0 in allo.reduction(Q):
            out_AB[i0, j0] += A[i0, k0] * B[k0, j0]


def mm2[
    DType: (float32, int32), R: int32, S: int32, T: int32
](C: "DType[R, S]", D: "DType[S, T]", out_CD: "DType[R, T]"):
    for i1, j1 in allo.grid(R, T, name="mm2"):
        for k1 in allo.reduction(S):
            out_CD[i1, j1] += C[i1, k1] * D[k1, j1]


def mm3[
    DType: (float32, int32), P: int32, R: int32, T: int32
](out_AB: "DType[P, R]", out_CD: "DType[R, T]", out_ABC: "DType[P, T]"):
    for i2, j2 in allo.grid(P, T, name="mm3"):
        for k2 in allo.reduction(R):
            out_ABC[i2, j2] += out_AB[i2, k2] * out_CD[k2, j2]


def kernel_3mm[
    DType: (float32, int32), P: int32, Q: int32, R: int32, S: int32, T: int32
](
    A: "DType[P, Q]", B: "DType[Q, R]", C: "DType[R, S]", D: "DType[S, T]"
) -> "DType[P, T]":
    out_AB: DType[P, R] = 0
    out_CD: DType[R, T] = 0
    output: DType[P, T] = 0
    mm1[DType, P, Q, R](A, B, out_AB)
    mm2[DType, R, S, T](C, D, out_CD)
    mm3[DType, P, R, T](out_AB, out_CD, output)
    return output


def three_mm(concrete_type, p, r, q, t, s):
    sch0 = allo.customize(mm1, instantiate=[concrete_type, p, q, r])
    sch0.reorder("k0", "j0")
    sch0.buffer_at(sch0.out_AB, axis="i0")
    sch0.pipeline("k0")
    i0 = sch0.get_loops("mm1")["mm1"]["i0"]
    sch0.dataflow(i0)

    sch1 = allo.customize(mm2, instantiate=[concrete_type, r, s, t])
    sch1.reorder("k1", "j1")
    sch1.buffer_at(sch1.out_CD, axis="i1")
    sch1.pipeline("k1")
    i1 = sch1.get_loops("mm2")["mm2"]["i1"]
    sch1.dataflow(i1)

    sch2 = allo.customize(mm3, instantiate=[concrete_type, p, r, t])
    sch2.reorder("k2", "j2")
    sch2.buffer_at(sch2.out_ABC, axis="i2")
    sch2.pipeline("k2")
    i2 = sch2.get_loops("mm3")["mm3"]["i2"]
    sch2.dataflow(i2)

    sch = allo.customize(kernel_3mm, instantiate=[concrete_type, p, q, r, s, t])
    sch.compose(sch0)
    sch.compose(sch1)
    sch.compose(sch2)

    # sch.to(sch.out_AB, "mm3")
    # sch.to(sch.out_CD, "mm3")

    sch.partition(sch.B, dim=2)
    sch.partition(sch.C, dim=2)
    sch.partition(sch.out_CD, dim=0)

    sch.dataflow("mm1")
    sch.dataflow("mm2")
    sch.dataflow("mm3")

    return sch


def test_three_mm():
    # read problem size settings
    setting_path = os.path.join(os.path.dirname(__file__), "psize.json")
    with open(setting_path, "r") as fp:
        psize = json.load(fp)
    # for CI test we use small problem size
    test_psize = "small"
    P = psize["three_mm"][test_psize]["P"]
    R = psize["three_mm"][test_psize]["R"]
    Q = psize["three_mm"][test_psize]["Q"]
    T = psize["three_mm"][test_psize]["T"]
    S = psize["three_mm"][test_psize]["S"]

    concrete_type = float32
    sch = three_mm(concrete_type, P, R, Q, T, S)
    mod = sch.build()
    A = np.random.randint(-10, 10, (P, Q)).astype(np.float32)
    B = np.random.randint(-10, 10, (Q, R)).astype(np.float32)
    C = np.random.randint(-10, 10, (R, S)).astype(np.float32)
    D = np.random.randint(-10, 10, (S, T)).astype(np.float32)
    out = mod(A, B, C, D)
    out_ref = three_mm_np(A, B, C, D)
    np.testing.assert_allclose(out, out_ref, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__])
