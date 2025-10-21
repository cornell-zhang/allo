# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import json
import pytest
import allo
import numpy as np
from allo.ir.types import int32, float32
import allo.ir.types as T


def trmm_np(A, B, alpha):
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            for k in range(i + 1, A.shape[0]):
                B[i, j] += A[k, i] * B[k, j]
            B[i, j] *= alpha
    return B


def S0[T: (float32, int32), M, N](A: "T[M, M]", B: "T[M, N]"):
    for i1, j1 in allo.grid(M, N, name="update"):
        for k1 in allo.reduction(M):
            if k1 > i1:
                B[i1, j1] += A[k1, i1] * B[k1, j1]


def S1[T: (float32, int32), M, N](B: "T[M, N]"):
    for i0, j0 in allo.grid(M, N, name="mul"):
        B[i0, j0] = B[i0, j0] * alpha


def kernel_trmm[T: (float32, int32), M, N](A: "T[M, M]", B: "T[M, N]"):
    S0[T, M, N](A, B)
    S1[T, M, N](B)


def top_trmm(concrete_type, m, n, alpha=1.5):
    factor = 20
    s0 = allo.customize(S0, instantiate=[concrete_type, m, n])
    s0.partition(s0.B, dim=2, partition_type=2, factor=factor)  # cyclic
    s0.reorder("k1", "j1")
    # s0.buffer_at(s0.B, "i1") # broke functional test, should load from s0.buffer
    s0.pipeline("j1")
    s0.unroll("j1", factor=factor)

    s1 = allo.customize(S1, instantiate=[concrete_type, m, n])
    s1.pipeline("j0")
    s1.unroll("j0", factor=factor)

    s = allo.customize(kernel_trmm, instantiate=[concrete_type, m, n])
    s.compose(s0)
    s.compose(s1)
    return s


def test_trmm():
    # read problem size settings
    setting_path = os.path.join(os.path.dirname(__file__), "psize.json")
    with open(setting_path, "r") as fp:
        psize = json.load(fp)
    # for CI test we use small problem size
    test_psize = "small"
    M = psize["trmm"][test_psize]["M"]
    N = psize["trmm"][test_psize]["N"]
    concrete_type = float32
    alpha = 1.5
    s = top_trmm(concrete_type, M, N, alpha)
    A = np.random.randint(0, 10, size=(M, M)).astype(np.float32)
    B = np.random.randint(0, 10, size=(M, N)).astype(np.float32)
    B_golden = B.copy()
    B_golden = trmm_np(A, B_golden, alpha)
    mod = s.build()
    mod(A, B)
    np.testing.assert_allclose(B, B_golden, atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__])
