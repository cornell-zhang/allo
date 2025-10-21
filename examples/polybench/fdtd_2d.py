# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import json
import pytest
import allo
import numpy as np
from allo.ir.types import int32, float32
import allo.ir.types as T


def fdtd_2d_np(ex, ey, hz, fict):
    NX, NY = ex.shape
    TMAX = fict.shape[0]

    for t in range(TMAX):
        for j in range(NY):
            ey[0, j] = fict[t]
        for i in range(1, NX):
            for j in range(NY):
                ey[i, j] = ey[i, j] - 0.5 * (hz[i, j] - hz[i - 1, j])
        for i in range(NX):
            for j in range(1, NY):
                ex[i, j] = ex[i, j] - 0.5 * (hz[i, j] - hz[i, j - 1])
        for i in range(NX - 1):
            for j in range(NY - 1):
                hz[i, j] = hz[i, j] - 0.7 * (
                    ex[i, j + 1] - ex[i, j] + ey[i + 1, j] - ey[i, j]
                )
    return ex, ey, hz, fict


def kernel_fdtd_2d[
    T: (float32, int32), Nx: int32, Ny: int32, Tmax: int32
](ex: "T[Nx, Ny]", ey: "T[Nx, Ny]", hz: "T[Nx, Ny]", fict: "T[Tmax]",):
    for m in allo.grid(Tmax):
        for j in allo.grid(Ny):
            ey[0, j] = fict[m]

        for i in range(1, Nx):
            for j in allo.grid(Ny):
                ey[i, j] = ey[i, j] - 0.5 * (hz[i, j] - hz[i - 1, j])

        for i in range(Nx):
            for j in range(1, Ny):
                ex[i, j] = ex[i, j] - 0.5 * (hz[i, j] - hz[i, j - 1])

        for i in allo.grid(Nx - 1):
            for j in allo.grid(Ny - 1):
                hz[i, j] = hz[i, j] - 0.7 * (
                    ex[i, j + 1] - ex[i, j] + ey[i + 1, j] - ey[i, j]
                )


def top_fdtd_2d(concrete_type, Nx, Ny, Tmax):
    s = allo.customize(kernel_fdtd_2d, instantiate=[concrete_type, Nx, Ny, Tmax])
    return s.build()


def test_fdtd_2d():
    # read problem size settings
    setting_path = os.path.join(os.path.dirname(__file__), "psize.json")
    with open(setting_path, "r") as fp:
        psize = json.load(fp)
    # for CI test we use small problem size
    test_psize = "small"
    Nx = psize["fdtd_2d"][test_psize]["Nx"]
    Ny = psize["fdtd_2d"][test_psize]["Ny"]
    Tmax = psize["fdtd_2d"][test_psize]["Tmax"]
    concrete_type = float32
    mod = top_fdtd_2d(concrete_type, Nx, Ny, Tmax)

    ex = np.random.randint(-10, 10, size=(Nx, Ny)).astype(np.float32)
    ey = np.random.randint(-10, 10, size=(Nx, Ny)).astype(np.float32)
    hz = np.random.randint(-10, 10, size=(Nx, Ny)).astype(np.float32)
    fict = np.random.randint(-10, 10, size=(Tmax,)).astype(np.float32)
    ex_golden = ex.copy()
    ey_golden = ey.copy()
    hz_golden = hz.copy()
    fict_golden = fict.copy()
    ex_golden, ey_golden, hz_golden, fict_golden = fdtd_2d_np(
        ex_golden, ey_golden, hz_golden, fict_golden
    )
    mod(ex, ey, hz, fict)
    np.testing.assert_allclose(ex, ex_golden, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(ey, ey_golden, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(hz, hz_golden, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(fict, fict_golden, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__])
