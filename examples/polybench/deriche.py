# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import json
import pytest
import allo
import numpy as np
from allo.ir.types import int32, float32, index
import allo.ir.types as T
import math


def SCALAR_VAL(x):
    return x


def POW_FUN(x, y):
    return x**y


def EXP_FUN(x):
    return math.exp(x)


def deriche_np(imgIn, imgOut, y1, y2, a1, a2, a3, a4, a5, a6, a7, a8, b1, b2, c1, c2):
    W, H = imgIn.shape
    for i in range(W):
        ym1 = 0.0
        ym2 = 0.0
        xm1 = 0.0
        for j in range(H):
            y1[i][j] = a1 * imgIn[i][j] + a2 * xm1 + b1 * ym1 + b2 * ym2
            xm1 = imgIn[i][j]
            ym2 = ym1
            ym1 = y1[i][j]

    for i in range(W):
        yp1, yp2, xp1, xp2 = 0.0, 0.0, 0.0, 0.0
        for j in range(H - 1, -1, -1):
            y2[i][j] = a3 * xp1 + a4 * xp2 + b1 * yp1 + b2 * yp2
            xp2 = xp1
            xp1 = imgIn[i][j]
            yp2 = yp1
            yp1 = y2[i][j]

    for i in range(W):
        for j in range(H):
            imgOut[i][j] = c1 * (y1[i][j] + y2[i][j])

    for j in range(H):
        tm1, ym1, ym2 = 0.0, 0.0, 0.0
        for i in range(W):
            y1[i][j] = a5 * imgOut[i][j] + a6 * tm1 + b1 * ym1 + b2 * ym2
            tm1 = imgOut[i][j]
            ym2 = ym1
            ym1 = y1[i][j]

    for j in range(H):
        tp1, tp2, yp1, yp2 = 0.0, 0.0, 0.0, 0.0
        for i in range(W - 1, -1, -1):
            y2[i][j] = a7 * tp1 + a8 * tp2 + b1 * yp1 + b2 * yp2
            tp2 = tp1
            tp1 = imgOut[i][j]
            yp2 = yp1
            yp1 = y2[i][j]

    for i in range(W):
        for j in range(H):
            imgOut[i][j] = c2 * (y1[i][j] + y2[i][j])
    return imgIn, imgOut, y1, y2


# Global constants for kernel functions
a1 = a2 = a3 = a4 = a5 = a6 = a7 = a8 = b1 = b2 = c1 = c2 = 0.0


def kernel_deriche[
    T: (float32, int32), W: int32, H: int32
](imgIn: "T[W, H]", imgOut: "T[W, H]", y1: "T[W, H]", y2: "T[W, H]"):
    for i in range(W):
        ym1: T = 0.0
        ym2: T = 0.0
        xm1: T = 0.0
        for j in range(H):
            y1[i, j] = a1 * imgIn[i, j] + a2 * xm1 + b1 * ym1 + b2 * ym2
            xm1 = imgIn[i, j]
            ym2 = ym1
            ym1 = y1[i, j]

    for i in range(W):
        yp1: T = 0.0
        yp2: T = 0.0
        xp1: T = 0.0
        xp2: T = 0.0
        for j_inv in range(H):
            j: index = H - 1 - j_inv
            y2[i, j] = a3 * xp1 + a4 * xp2 + b1 * yp1 + b2 * yp2
            xp2 = xp1
            xp1 = imgIn[i, j]
            yp2 = yp1
            yp1 = y2[i, j]

    for i in range(W):
        for j in range(H):
            imgOut[i, j] = c1 * (y1[i, j] + y2[i, j])

    for j in range(H):
        tm1: T = 0.0
        ym1: T = 0.0
        ym2: T = 0.0
        for i in range(W):
            y1[i, j] = a5 * imgOut[i, j] + a6 * tm1 + b1 * ym1 + b2 * ym2
            tm1 = imgOut[i, j]
            ym2 = ym1
            ym1 = y1[i, j]

    for j in range(H):
        tp1: T = 0.0
        tp2: T = 0.0
        yp1: T = 0.0
        yp2: T = 0.0
        # for i in range(W-1, -1, -1):
        for i_inv in range(W):
            i: index = W - 1 - i_inv
            y2[i, j] = a7 * tp1 + a8 * tp2 + b1 * yp1 + b2 * yp2
            tp2 = tp1
            tp1 = imgOut[i, j]
            yp2 = yp1
            yp1 = y2[i, j]

    for i in range(W):
        for j in range(H):
            imgOut[i, j] = c2 * (y1[i, j] + y2[i, j])


def deriche(
    concrete_type,
    w,
    h,
    a1_val,
    a2_val,
    a3_val,
    a4_val,
    a5_val,
    a6_val,
    a7_val,
    a8_val,
    b1_val,
    b2_val,
    c1_val,
    c2_val,
):
    global a1, a2, a3, a4, a5, a6, a7, a8, b1, b2, c1, c2
    a1, a2, a3, a4 = a1_val, a2_val, a3_val, a4_val
    a5, a6, a7, a8 = a5_val, a6_val, a7_val, a8_val
    b1, b2 = b1_val, b2_val
    c1, c2 = c1_val, c2_val

    s0 = allo.customize(kernel_deriche, instantiate=[concrete_type, w, h])
    return s0.build()


def test_deriche():
    # read problem size settings
    setting_path = os.path.join(os.path.dirname(__file__), "psize.json")
    with open(setting_path, "r") as fp:
        psize = json.load(fp)
    # for CI test we use small problem size
    test_psize = "small"
    W = psize["deriche"][test_psize]["W"]
    H = psize["deriche"][test_psize]["H"]
    alpha = 0.25
    k = (
        (SCALAR_VAL(1.0) - EXP_FUN(-alpha))
        * (SCALAR_VAL(1.0) - EXP_FUN(-alpha))
        / (
            SCALAR_VAL(1.0)
            + SCALAR_VAL(2.0) * alpha * EXP_FUN(-alpha)
            - EXP_FUN(SCALAR_VAL(2.0) * alpha)
        )
    )
    a1 = a5 = k
    a2 = a6 = k * EXP_FUN(-alpha) * (alpha - SCALAR_VAL(1.0))
    a3 = a7 = k * EXP_FUN(-alpha) * (alpha + SCALAR_VAL(1.0))
    a4 = a8 = -k * EXP_FUN(SCALAR_VAL(-2.0) * alpha)
    b1 = POW_FUN(SCALAR_VAL(2.0), -alpha)
    b2 = -EXP_FUN(SCALAR_VAL(-2.0) * alpha)
    c1 = c2 = 1

    imgIn = np.random.rand(W, H).astype(np.float32)
    imgOut = np.zeros_like(imgIn)
    y1 = np.zeros_like(imgIn)
    y2 = np.zeros_like(imgIn)

    imgOut_ref = np.zeros_like(imgIn)
    y1_ref = np.zeros_like(imgIn)
    y2_ref = np.zeros_like(imgIn)

    imgIn_ref, imgOut_ref, y1_ref, y2_ref = deriche_np(
        imgIn.copy(),
        imgOut_ref,
        y1_ref,
        y2_ref,
        a1,
        a2,
        a3,
        a4,
        a5,
        a6,
        a7,
        a8,
        b1,
        b2,
        c1,
        c2,
    )

    mod = deriche(float32, W, H, a1, a2, a3, a4, a5, a6, a7, a8, b1, b2, c1, c2)
    mod(imgIn, imgOut, y1, y2)

    np.testing.assert_allclose(imgOut_ref, imgOut, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(y1_ref, y1, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(y2_ref, y2, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__])
