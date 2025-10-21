# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import json
import pytest
import allo
import numpy as np
from allo.ir.types import int32, float32
import allo.ir.types as T


def durbin_np(r, y):
    z = np.zeros_like(y)
    N = r.shape[0]
    y[0] = -r[0]
    beta = 1.0
    alpha = -r[0]
    for k in range(1, N):
        beta = (1 - alpha * alpha) * beta
        sum_ = 0.0
        for i in range(k):
            sum_ = sum_ + r[k - i - 1] * y[i]
        alpha = -1.0 * (r[k] + sum_)
        # alpha = alpha / beta
        for i in range(k):
            z[i] = y[i] + alpha * y[k - i - 1]
        for i in range(k):
            y[i] = z[i]
        y[k] = alpha
    return y


def kernel_durbin[T: (float32, int32), N: int32](r: "T[N]", y: "T[N]"):
    y[0] = -r[0]
    beta: T = 1.0
    alpha: T = -r[0]

    for k in range(1, N):
        beta = (1 - alpha * alpha) * beta
        sum_: T = 0.0

        z: T[N] = 0.0
        for i in range(k):
            sum_ = sum_ + r[k - i - 1] * y[i]

        alpha = -1.0 * (r[k] + sum_)
        # alpha = alpha / beta # unstable

        for i in range(k):
            z[i] = y[i] + alpha * y[k - i - 1]

        for i in range(k):
            y[i] = z[i]

        y[k] = alpha


def durbin(concrete_type, n):
    s = allo.customize(kernel_durbin, instantiate=[concrete_type, n])
    return s.build()


def test_durbin():
    # read problem size settings
    setting_path = os.path.join(os.path.dirname(__file__), "psize.json")
    with open(setting_path, "r") as fp:
        psize = json.load(fp)
    # for CI test we use small problem size
    test_psize = "small"
    N = psize["durbin"][test_psize]["N"]
    concrete_type = float32
    r = np.random.randint(1, 10, size=(N,)).astype(np.float32)
    y = np.random.randint(1, 10, size=(N,)).astype(np.float32)
    y_golden = y.copy()
    y_golden = durbin_np(r, y_golden)
    mod = durbin(concrete_type, N)
    mod(r.copy(), y)
    np.testing.assert_allclose(y, y_golden, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__])
