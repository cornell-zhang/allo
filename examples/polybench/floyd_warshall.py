# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import json
import pytest
import allo
import numpy as np
from allo.ir.types import int32, float32
import allo.ir.types as T


def floyd_warshall_np(path):
    N = path.shape[0]
    for k in range(N):
        for i in range(N):
            for j in range(N):
                path[i, j] = min(path[i, j], path[i, k] + path[k, j])
    return path


def kernel_floyd_warshall[T: (float32, int32), N: int32](path: "T[N, N]"):
    for k, i, j in allo.grid(N, N, N):
        path_: T = path[i, k] + path[k, j]
        if path[i, j] >= path_:
            path[i, j] = path_


def floyd_warshall(concrete_type, N):
    s0 = allo.customize(kernel_floyd_warshall, instantiate=[concrete_type, N])
    return s0.build()


def test_floyd_warshall():
    # read problem size settings
    setting_path = os.path.join(os.path.dirname(__file__), "psize.json")
    with open(setting_path, "r") as fp:
        psize = json.load(fp)
    # for CI test we use small problem size
    test_psize = "small"
    N = psize["floyd_warshall"][test_psize]["N"]
    concrete_type = float32
    mod = floyd_warshall(concrete_type, N)
    path = np.random.rand(N, N).astype(np.float32)
    path_ref = path.copy()
    path_ref = floyd_warshall_np(path_ref)
    mod(path)
    np.testing.assert_allclose(path, path_ref, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__])
