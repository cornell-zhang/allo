import allo
import numpy as np
from allo.ir.types import float32, int32


def top_deriche(
    size="mini",
    alpha=0.25,
    k=0.1,
    a1=0.1,
    a2=0.1,
    a3=0.1,
    a4=0.1,
    a5=0.1,
    a6=0.1,
    a7=0.1,
    a8=0.1,
    b1=0.1,
    b2=0.1,
    c1=0.1,
    c2=0.1,
):
    if size == "mini" or size is None:
        W = 64
        H = 64
    elif size == "small":
        W = 192
        H = 128

    elif size == "medium":
        W = 720
        H = 480

    def kernel_deriche(ImageIn: float32[W, H], ImageOut: float32[W, H]):
        cst_neg_1: int32 = -1
        # Placeholder initialization
        y1: float32[W, H] = 0.0
        y2: float32[W, H] = 0.0

        # Compute y1 and y2
        for i in allo.grid(W):
            y1_d1: float32 = 0.0
            y1_d2: float32 = 0.0
            x_d1: float32 = 0.0
            for j in allo.grid(H):
                y1[i, j] = a1 * ImageIn[i, j] + a2 * x_d1 + b1 * y1_d1 + b2 * y1_d2
                x_d1 = ImageIn[i, j]
                y1_d2 = y1_d1
                y1_d1 = y1[i, j]

        for i in allo.grid(W):
            y2_d1: float32 = 0.0
            y2_d2: float32 = 0.0
            x_d1: float32 = 0.0
            x_d2: float32 = 0.0
            for j in range(H - 1, cst_neg_1, cst_neg_1):
                y2[i, j] = a3 * x_d1 + a4 * x_d2 + b1 * y2_d1 + b2 * y2_d2
                x_d2 = x_d1
                x_d1 = ImageIn[i, j]
                y2_d2 = y2_d1
                y2_d1 = y2[i, j]

        for i, j in allo.grid(W, H):
            ImageOut[i, j] = c1 * (y1[i, j] + y2[i, j])

        for j in allo.grid(H):
            ImageOut_d1: float32 = 0.0
            y1_d1: float32 = 0.0
            y1_d2: float32 = 0.0
            for i in allo.grid(W):
                y1[i, j] = (
                    a5 * ImageOut[i, j] + a6 * ImageOut_d1 + b1 * y1_d1 + b2 * y1_d2
                )
                ImageOut_d1 = ImageOut[i, j]
                y1_d2 = y1_d1
                y1_d1 = y1[i, j]

        for j in allo.grid(H):
            ImageOut_d1: float32 = 0.0
            ImageOut_d2: float32 = 0.0
            y2_d1: float32 = 0.0
            y2_d2: float32 = 0.0
            for i in range(W - 1, cst_neg_1, cst_neg_1):
                y2[i, j] = a7 * ImageOut_d1 + a8 * ImageOut_d2 + b1 * y2_d1 + b2 * y2_d2
                ImageOut_d2 = ImageOut_d1
                ImageOut_d1 = ImageOut[i, j]
                y2_d2 = y2_d1
                y2_d1 = y2[i, j]

        for i, j in allo.grid(W, H):
            ImageOut[i, j] = c2 * (y1[i, j] + y2[i, j])

    s0 = allo.customize(kernel_deriche)
    orig = s0.build("vhls")

    s1 = allo.customize(kernel_deriche)
    # s1.split("i", factor=2)
    # s1.reorder("j", "i.inner", "i.outer")
    # s1.pipeline("i.inner")
    opt = s1.build("vhls")

    return orig, opt


if __name__ == "__main__":
    orig, opt = top_deriche()
    from cedar.verify import verify_pair

    verify_pair(orig, opt, "deriche", liveout_vars="v1")
