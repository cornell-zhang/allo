import os
import shutil

import numpy as np

import allo
from allo.ir.types import int32


# Simple 4x4 GEMM kernel
def gemm(A: int32[4, 4], B: int32[4, 4]) -> int32[4, 4]:
    C: int32[4, 4] = 0
    for i, j, k in allo.grid(4, 4, 4):
        C[i, j] += A[i, k] * B[k, j]
    return C


def main():
    s = allo.customize(gemm)

    project = "gemm_xls_sw_emu.prj"
    if os.path.exists(project):
        shutil.rmtree(project)

    # Build XLS C++ backend with sw_emu enabled
    mod = s.build(target="xls", mode="sw_emu", project=project)

    print("Project directory:", os.path.abspath(project))
    print("Contents:")
    for name in sorted(os.listdir(project)):
        print("  ", name)

    # Run a small GEMM to exercise the sw_emu binary
    A = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [1, 0, 1, 0], [2, 2, 2, 2]], dtype=np.int32)
    B = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.int32)

    print("\nRunning sw_emu GEMM...")
    if hasattr(mod, "__call__"):
        C = mod(A, B)
        print("Result C (shape {}):".format(C.shape))
        print(C)
    else:
        print("sw_emu not available; only codegen was performed.")


if __name__ == "__main__":
    main()
