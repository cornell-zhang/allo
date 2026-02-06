import os
import sys
import json
import allo
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
import crs as crs_mod


def generate_sparse_crs(n, nnz, rng):
    """Generate a random sparse matrix in CRS format."""
    entries_per_row = nnz // n
    extra = nnz - entries_per_row * n

    values = rng.random(nnz).astype(np.float64)
    columns = np.zeros(nnz, dtype=np.int32)
    rows = np.zeros(n + 1, dtype=np.int32)

    idx = 0
    for i in range(n):
        count = entries_per_row + (1 if i < extra else 0)
        cols = np.sort(rng.choice(n, size=count, replace=False))
        for c in cols:
            columns[idx] = c
            idx += 1
        rows[i + 1] = idx

    return values, columns, rows


def test_spmv_crs(psize="small"):
    setting_path = os.path.join(os.path.dirname(__file__), "..", "..", "psize.json")
    with open(setting_path, "r") as fp:
        sizes = json.load(fp)
    params = sizes["spmv_crs"][psize]

    N = params["N"]
    NNZ = params["NNZ"]

    # Patch module constants
    crs_mod.N = N
    crs_mod.NNZ = NNZ

    rng = np.random.default_rng(42)

    values, columns, rows = generate_sparse_crs(N, NNZ, rng)
    vector = rng.random(N).astype(np.float64)

    # Python reference: dense matmul
    dense = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(rows[i], rows[i + 1]):
            dense[i, columns[j]] = values[j]
    expected = dense @ vector

    s = allo.customize(crs_mod.crs)
    mod = s.build()
    actual = mod(values, columns, rows, vector)

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    print("PASS!")


if __name__ == "__main__":
    test_spmv_crs("full")
