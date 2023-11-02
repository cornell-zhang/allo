# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
import numpy as np
from allo.ir.types import float32, int32


def top_adi(size="mini"):
    if size == "mini" or size is None:
        TSTEPS = 20
        N = 20
    elif size == "small":
        TSTEPS = 40
        N = 60
    elif size == "medium":
        TSTEPS = 100
        N = 200

    Nx = N
    Ny = Nx
    NT = TSTEPS

    Dx = 1.0 / Nx
    Dy = 1.0 / Ny
    DT = 1.0 / NT
    B1 = 2.0
    B2 = 1.0
    mu1 = B1 * DT / (Dx * Dx)
    mu2 = B2 * DT / (Dy * Dy)

    a = -1.0 * mu1 / 2.0
    b = 1.0 + mu1
    c = a
    d = -1.0 * mu2 / 2.0
    e = 1.0 + mu2
    f = d

    def kernel_adi(
        u: float32[Nx, Ny], v: float32[Nx, Ny], p: float32[Nx, Ny], q: float32[Nx, Ny]
    ):
        cst_neg_1: int32 = -1
        for m in allo.grid(NT):  # Outer loop
            # First block
            for i in range(1, Ny - 1):
                v[0, i] = 1.0
                # p[i, 0] = 0.0
                q[i, 0] = v[0, i]
                for j in range(1, Nx - 1):
                    p[i, j] = -1.0 * c / (a * p[i, j - 1] + b)
                    q[i, j] = (
                        -1.0 * d * u[j, i - 1]
                        + (1.0 + 2.0 * d) * u[j, i]
                        - f * u[j, i + 1]
                        - a * q[i, j - 1]
                    ) / (a * p[i, j - 1] + b)
                v[Nx - 1, i] = 1.0
                for j in range(Nx - 2, cst_neg_1, cst_neg_1):
                    v[j, i] = p[i, j] * v[j + 1, i] + q[i, j]

            # Second block
            for i in range(1, Nx - 1):
                u[i, 0] = 1.0
                # p[i, 0] = 0.0
                q[i, 0] = u[i, 0]
                for j in range(1, Ny - 1):
                    p[i, j] = -1.0 * f / (d * p[i, j - 1] + e)
                    q[i, j] = (
                        -1.0 * a * v[i - 1, j]
                        + (1.0 + 2 * a) * v[i, j]
                        - c * v[i + 1, j]
                        - d * q[i, j - 1]
                    ) / (d * p[i, j - 1] + e)
                u[i, Ny - 1] = 1.0
                for j in range(Ny - 2, cst_neg_1, cst_neg_1):
                    u[i, j] = p[i, j] * u[i, j + 1] + q[i, j]

    s0 = allo.customize(kernel_adi)
    orig = s0.build("vhls")

    s = allo.customize(kernel_adi)
    s.split("m", factor=2)
    s.reorder("m.inner", "m.outer")
    opt = s.build("vhls")
    return orig, opt


if __name__ == "__main__":
    orig, opt = top_adi()
    from cedar.verify import verify_pair

    verify_pair(orig, opt, "adi", liveout_vars="v3")
