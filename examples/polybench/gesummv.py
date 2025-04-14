# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import json
import pytest
import allo
import numpy as np
from allo.ir.types import int32, float32
import allo.ir.types as T


def gesummv_np(A, B, x, y, alpha, beta):
    tmp = np.zeros_like(y)
    for i in range(A.shape[0]):
        tmp[i] = 0.0
        y[i] = 0.0
        for j in range(A.shape[1]):
            tmp[i] += A[i, j] * x[j]
            y[i] += B[i, j] * x[j]
        y[i] = alpha * tmp[i] + beta * y[i]
    return y


def compute_tmp[
    T: (float32, int32), N: int32
](y_in: "T[N]", y_out: "T[N]", A: "T[N, N]", B: "T[N, N]", x: "T[N]", tmp: "T[N]"):
    tt: T[N] = 0.0
    yy: T[N]
    for i0 in allo.grid(N, name="load"):
        yy[i0] = y_in[i0]
    for i, j in allo.grid(N, N, name="tmp"):
        tt[i] += A[i, j] * x[j]
        yy[i] += B[i, j] * x[j]
    for i1 in allo.grid(N, name="store"):
        tmp[i1] = tt[i1]
        y_out[i1] = yy[i1]


def compute_y[T: (float32, int32), N: int32](y_in: "T[N]", y_out: "T[N]", tmp: "T[N]"):
    for i0 in allo.grid(N, name="load"):
        y_out[i0] = alpha * tmp[i0] + beta * y_in[i0]


def kernel_gesummv[
    T: (float32, int32), N: int32
](A: "T[N, N]", B: "T[N, N]", x: "T[N]", y: "T[N]"):
    y_init: T[N] = 0
    y_fifo: T[N]
    tmp: T[N]
    compute_tmp(y_init, y_fifo, A, B, x, tmp)
    compute_y(y_fifo, y, tmp)


def gesummv(concrete_type, N, alpha=0.1, beta=0.1):
    sch0 = allo.customize(compute_tmp, instantiate=[concrete_type, N])
    sch0.reorder("j", "i")

    sch1 = allo.customize(compute_y, instantiate=[concrete_type, N])
    sch1.pipeline("i0")

    sch = allo.customize(kernel_gesummv, instantiate=[concrete_type, N])
    sch.compose(sch0)
    sch.compose(sch1)

    return sch


def test_gesummv():
    # read problem size settings
    setting_path = os.path.join(os.path.dirname(__file__), "psize.json")
    with open(setting_path, "r") as fp:
        psize = json.load(fp)
    # for CI test we use small problem size
    test_psize = "small"
    N = psize["gesummv"][test_psize]["N"]
    concrete_type = float32
    # functional correctness test
    A = np.random.randint(10, size=(N, N)).astype(np.float32)
    B = np.random.randint(10, size=(N, N)).astype(np.float32)
    x = np.random.randint(10, size=(N)).astype(np.float32)
    y = np.zeros(N).astype(np.float32)
    y_ref = np.zeros(N).astype(np.float32)
    alpha, beta = 0.1, 0.1
    y_ref = gesummv_np(A, B, x, y_ref, alpha, beta)
    sch = gesummv(concrete_type, N, alpha, beta)
    mod = sch.build()
    mod(A, B, x, y)
    np.testing.assert_allclose(y, y_ref, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__])
