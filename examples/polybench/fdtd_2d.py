import allo
import numpy as np
from allo.ir.types import float32, int32


def top_fdtd_2d(size="tiny"):
    if size == "mini":
        Tmax = 20
        Nx = 20
        Ny = 30

    elif size == "small":
        Tmax = 40
        Nx = 60
        Ny = 80

    elif size == "medium":
        Tmax = 100
        Nx = 200
        Ny = 240

    elif size is None or size == "tiny":
        Tmax = 4
        Nx = 6
        Ny = 8

    def kernel_fdtd_2d(
        ex: float32[Nx, Ny],
        ey: float32[Nx, Ny],
        hz: float32[Nx, Ny],
        fict: float32[Tmax],
    ):
        const1: float32 = 0.5
        const2: float32 = 0.7

        for m in allo.grid(Tmax):
            for j in allo.grid(Ny):
                ey[0, j] = fict[m]

            for i in range(1, Nx):
                for j in allo.grid(Ny):
                    ey[i, j] = ey[i, j] - const1 * (hz[i, j] - hz[i - 1, j])

            for i in allo.grid(Nx):
                for j in range(1, Ny):
                    ex[i, j] = ex[i, j] - const1 * (hz[i, j] - hz[i, j - 1])

            for i in allo.grid(Nx - 1):
                for j in allo.grid(Ny - 1):
                    hz[i, j] = hz[i, j] - const2 * (
                        ex[i, j + 1] - ex[i, j] + ey[i + 1, j] - ey[i, j]
                    )

    s0 = allo.customize(kernel_fdtd_2d)
    orig = s0.build("vhls")

    s = allo.customize(kernel_fdtd_2d)
    opt = s.build("vhls")

    return orig, opt


if __name__ == "__main__":
    orig, opt = top_fdtd_2d()
    from cedar.verify import verify_pair

    verify_pair(orig, opt, "fdtd_2d", liveout_vars="v0,v1,v2")
