# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import json
import pytest
import allo
import numpy as np
from allo.ir.types import int32, float32
import allo.ir.types as T


def gramschmidt_np(A, Q, R):
    M = A.shape[0]
    N = A.shape[1]
    for k in range(N):
        nrm = 0.0
        for i in range(M):
            nrm += A[i, k] * A[i, k]
        # R[k, k] = np.sqrt(nrm)
        R[k, k] = nrm

        for i in range(M):
            Q[i, k] = A[i, k] / R[k, k]

        for j in range(k + 1, N):
            R[k, j] = 0.0
            for i in range(M):
                R[k, j] += Q[i, k] * A[i, j]

            for i in range(M):
                A[i, j] -= Q[i, k] * R[k, j]
    return A, Q, R


def kernel_gramschmidt[
    T: (float32, int32), M: int32, N: int32
](A: "T[M, N]", Q: "T[M, N]", R: "T[N, N]"):
    for k in range(N):
        nrm: T = 0.0
        for i in range(M):
            nrm += A[i, k] * A[i, k]
        # R[k, k] = allo.sqrt(nrm)
        R[k, k] = nrm

        for i in range(M):
            Q[i, k] = A[i, k] / R[k, k]

        for j in range(k + 1, N):
            R[k, j] = 0.0
            for i in range(M):
                R[k, j] += Q[i, k] * A[i, j]

            for i in range(M):
                A[i, j] -= Q[i, k] * R[k, j]


def gramschmidt(concrete_type, m, n):
    s0 = allo.customize(kernel_gramschmidt, instantiate=[concrete_type, m, n])
    return s0.build()


def test_gramschmidt():
    # read problem size settings
    setting_path = os.path.join(os.path.dirname(__file__), "psize.json")
    with open(setting_path, "r") as fp:
        psize = json.load(fp)
    # for CI test we use small problem size
    test_psize = "small"

    # generate input data
    M = psize["gramschmidt"][test_psize]["M"]
    N = psize["gramschmidt"][test_psize]["N"]
    A = np.random.randint(0, 10, size=(M, N)).astype(np.float32)
    Q = np.zeros((N, N), dtype=np.float32)
    R = np.zeros((N, N), dtype=np.float32)

    # run reference
    A_ref = A.copy()
    Q_ref = Q.copy()
    R_ref = R.copy()
    A_ref, Q_ref, R_ref = gramschmidt_np(A_ref, Q_ref, R_ref)

    # run allo
    A_opt = A.copy()
    Q_opt = Q.copy()
    R_opt = R.copy()
    gramschmidt(float32, M, N)(A_opt, Q_opt, R_opt)

    # verify
    np.testing.assert_allclose(A_ref, A_opt, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(Q_ref, Q_opt, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(R_ref, R_opt, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__])
