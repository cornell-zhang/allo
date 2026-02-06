import os
import sys
import json
import allo
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
import stencil2d as stencil2d_mod
import stencil3d as stencil3d_mod


def test_stencil2d(psize="small"):
    setting_path = os.path.join(os.path.dirname(__file__), "..", "psize.json")
    with open(setting_path, "r") as fp:
        sizes = json.load(fp)
    params = sizes["stencil2d"][psize]

    row_size = params["row_size"]
    col_size = params["col_size"]

    # Patch module constants
    stencil2d_mod.row_size = row_size
    stencil2d_mod.col_size = col_size

    s = allo.customize(stencil2d_mod.stencil2d)
    mod = s.build(target="llvm")

    np.random.seed(42)
    f_size = 9
    np_orig = np.random.randint(0, 100, (row_size, col_size)).astype(np.int32)
    np_filter = np.random.randint(0, 10, (f_size,)).astype(np.int32)

    np_sol = mod(np_orig, np_filter)

    # Python reference
    ref_sol = np.zeros((row_size, col_size), dtype=np.int32)
    for r in range(row_size - 2):
        for c in range(col_size - 2):
            temp = 0
            for k1 in range(3):
                for k2 in range(3):
                    temp += np_filter[k1 * 3 + k2] * np_orig[r + k1, c + k2]
            ref_sol[r, c] = temp

    np.testing.assert_allclose(np_sol, ref_sol, rtol=1e-5, atol=1e-5)
    print("Stencil2D PASS!")


def test_stencil3d(psize="small"):
    setting_path = os.path.join(os.path.dirname(__file__), "..", "psize.json")
    with open(setting_path, "r") as fp:
        sizes = json.load(fp)
    params = sizes["stencil3d"][psize]

    height_size = params["height_size"]
    col_size = params["col_size"]
    row_size = params["row_size"]

    # Patch module constants
    stencil3d_mod.height_size = height_size
    stencil3d_mod.col_size = col_size
    stencil3d_mod.row_size = row_size

    s = allo.customize(stencil3d_mod.stencil3d)
    mod = s.build(target="llvm")

    np.random.seed(42)
    np_C = np.random.randint(1, 5, size=2).astype(np.int32)
    np_orig = np.random.randint(0, 100, (row_size, col_size, height_size)).astype(np.int32)

    np_sol = mod(np_C, np_orig)

    # Python reference
    ref_sol = np.zeros((row_size, col_size, height_size), dtype=np.int32)

    # Boundary: top/bottom height planes
    for j in range(col_size):
        for k in range(row_size):
            ref_sol[k, j, 0] = np_orig[k, j, 0]
            ref_sol[k, j, height_size - 1] = np_orig[k, j, height_size - 1]

    # Boundary: front/back col planes
    for i in range(height_size - 1):
        for k in range(row_size):
            ref_sol[k, 0, i+1] = np_orig[k, 0, i+1]
            ref_sol[k, col_size - 1, i+1] = np_orig[k, col_size - 1, i+1]

    # Boundary: left/right row planes
    for j in range(col_size - 2):
        for i in range(height_size - 2):
            ref_sol[0, j+1, i+1] = np_orig[0, j+1, i+1]
            ref_sol[row_size - 1, j+1, i+1] = np_orig[row_size - 1, j+1, i+1]

    # Interior stencil
    for i in range(height_size - 2):
        for j in range(col_size - 2):
            for k in range(row_size - 2):
                s0 = np_orig[k+1, j+1, i+1]
                s1 = (np_orig[k+1, j+1, i+2] +
                      np_orig[k+1, j+1, i] +
                      np_orig[k+1, j+2, i+1] +
                      np_orig[k+1, j, i+1] +
                      np_orig[k+2, j+1, i+1] +
                      np_orig[k, j+1, i+1])
                ref_sol[k+1, j+1, i+1] = s0 * np_C[0] + s1 * np_C[1]

    np.testing.assert_allclose(np_sol, ref_sol, rtol=1e-5, atol=1e-5)
    print("Stencil3D PASS!")


if __name__ == "__main__":
    test_stencil2d("full")
    test_stencil3d("full")
