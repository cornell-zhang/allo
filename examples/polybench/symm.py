# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import json
import pytest
import allo
import numpy as np
from allo.ir.types import int32, float32
import allo.ir.types as T


def symm_np(A, B, C, alpha, beta, M, N):
    for i in range(M):
        for j in range(N):
            sum_ = 0.0
            for k in range(i):
                C[k, j] = C[k, j] + alpha * B[i, j] * A[i, k]
                sum_ += B[k, j] * A[i, k]
            C[i, j] = beta * C[i, j] + alpha * B[i, j] * A[i, i] + alpha * sum_
    return C


def compute_sum[
    T: (float32, int32), M: int32, N: int32
](A: "T[M, M]", B: "T[M, N]", summ: "T[M, N]"):
    for i1, j1 in allo.grid(M, N, name="sum"):
        for k1 in allo.reduction(M, name="k1"):
            if k1 < i1:
                summ[i1, j1] += B[k1, j1] * A[i1, k1]


def update_C[
    T: (float32, int32), M: int32, N: int32
](A: "T[M, M]", B: "T[M, N]", summ: "T[M, N]", C: "T[M, N]",):
    for i in range(M):
        for k in range(i):  # pipeline
            for j in range(N):  # unroll
                C[k, j] = C[k, j] + alpha * B[i, j] * A[i, k]
        for j1 in range(N):
            C[i, j1] = (
                beta * C[i, j1] + alpha * B[i, j1] * A[i, i] + alpha * summ[i, j1]
            )


def kernel_symm[
    T: (float32, int32), M: int32, N: int32
](A0: "T[M, M]", A1: "T[M, M]", B0: "T[M, N]", B1: "T[M, N]", C: "T[M, N]"):
    # dataflow
    summ: T[M, N] = 0
    compute_sum(A0, B0, summ)
    update_C(A1, B1, summ, C)


def symm(concrete_type, M, N, alpha=1.5, beta=1.2):
    s0 = allo.customize(update_C, instantiate=[concrete_type, M, N])
    s0.pipeline("k")
    s0.partition(s0.C, dim=2)
    s0.partition(s0.B, dim=2)
    s0.pipeline("j1")
    s0.partition(s0.summ, dim=2)

    s1 = allo.customize(compute_sum, instantiate=[concrete_type, M, N])
    s1.reorder("k1", "j1")
    s1.buffer_at(s1.summ, "i1")
    s1.partition(s1.B, dim=2, partition_type=2, factor=24)
    s1.pipeline("j1")
    s1.unroll("j1", factor=24)

    sch = allo.customize(kernel_symm, instantiate=[concrete_type, M, N])
    sch.compose(s0)
    sch.compose(s1)
    return sch


def test_symm():
    # read problem size settings
    setting_path = os.path.join(os.path.dirname(__file__), "psize.json")
    with open(setting_path, "r") as fp:
        psize = json.load(fp)
    # for CI test we use small problem size
    test_psize = "small"
    M = psize["symm"][test_psize]["M"]
    N = psize["symm"][test_psize]["N"]
    alpha = 1.5
    beta = 1.2
    sch = symm(float32, M, N, alpha, beta)
    mod = sch.build()
    # functional testing
    A = np.random.randint(10, size=(M, M)).astype(np.float32)
    B = np.random.randint(10, size=(M, N)).astype(np.float32)
    C = np.random.randint(10, size=(M, N)).astype(np.float32)
    C_golden = C.copy()
    C_golden = symm_np(A, B, C_golden, alpha, beta, M, N)
    mod = sch.build()
    mod(A.copy(), A.copy(), B.copy(), B.copy(), C)
    np.testing.assert_allclose(C, C_golden, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__])
