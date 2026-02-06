# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import json
import allo
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
import ellpack as ellpack_mod


def generate_sparse_ellpack(n, l, rng):
    """Generate a random sparse matrix in ELLPACK format."""
    values = np.zeros(n * l, dtype=np.float64)
    columns = np.zeros(n * l, dtype=np.int32)

    for i in range(n):
        cols = np.sort(rng.choice(n, size=l, replace=False))
        for j in range(l):
            values[i * l + j] = rng.random()
            columns[i * l + j] = cols[j]

    return values, columns


def test_spmv_ellpack(psize="small"):
    setting_path = os.path.join(os.path.dirname(__file__), "..", "..", "psize.json")
    with open(setting_path, "r") as fp:
        sizes = json.load(fp)
    params = sizes["spmv_ellpack"][psize]

    N = params["N"]
    L = params["L"]

    # Patch module constants
    ellpack_mod.N = N
    ellpack_mod.L = L

    rng = np.random.default_rng(42)

    values, columns = generate_sparse_ellpack(N, L, rng)
    vector = rng.random(N).astype(np.float64)

    # Python reference: dense matmul
    dense = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(L):
            dense[i, columns[i * L + j]] = values[i * L + j]
    expected = dense @ vector

    s = allo.customize(ellpack_mod.ellpack)
    mod = s.build()
    observed = mod(values, columns, vector)

    np.testing.assert_allclose(observed, expected, rtol=1e-5, atol=1e-5)
    print("PASS!")


if __name__ == "__main__":
    test_spmv_ellpack("full")
