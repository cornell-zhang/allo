# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# transpose_fft.py

import allo
from allo.ir.types import float32, int32, float32, int32, index
import math
import numpy as np

def cmplx_M_x(a_x: float32, a_y: float32, b_x: float32, b_y: float32) -> float32:
    return a_x * b_x - a_y * b_y

def cmplx_M_y(a_x: float32, a_y: float32, b_x: float32, b_y: float32) -> float32:
    return a_x * b_y + a_y * b_x

def cmplx_MUL_x(a_x: float32, a_y: float32, b_x: float32, b_y: float32) -> float32:
    return a_x * b_x - a_y * b_y

def cmplx_MUL_y(a_x: float32, a_y: float32, b_x: float32, b_y: float32) -> float32:
    return a_x * b_y + a_y * b_x

def cmplx_mul_x(a_x: float32, a_y: float32, b_x: float32, b_y: float32) -> float32:
    return a_x * b_x - a_y * b_y

def cmplx_mul_y(a_x: float32, a_y: float32, b_x: float32, b_y: float32) -> float32:
    return a_x * b_y + a_y * b_x

def cmplx_add_x(a_x: float32, b_x: float32) -> float32:
    return a_x + b_x

def cmplx_add_y(a_y: float32, b_y: float32) -> float32:
    return a_y + b_y

def cmplx_sub_x(a_x: float32, b_x: float32) -> float32:
    return a_x - b_x

def cmplx_sub_y(a_y: float32, b_y: float32) -> float32:
    return a_y - b_y

def cm_fl_mul_x(a_x: float32, b: float32) -> float32:
    return b * a_x

def cm_fl_mul_y(a_y: float32, b: float32) -> float32:
    return b * a_y

def twiddles8(a_x: float32[8], a_y: float32[8], i: int32, n: int32):
    PI: float32 = 3.1415926535
    reversed8: int32[8] = [0, 4, 2, 6, 1, 5, 3, 7]

    for j in range(1, 8):
        # BUG: WITH COS AND SIN NOT BEING ABLE TO DEAL WITH float32
        phi: float32 = (-2 * PI * reversed8[j]/n)*i
        phi_x: float32 = allo.cos(phi)
        phi_y: float32 = allo.sin(phi)
        tmp_1: float32 = a_x[j]
        a_x[j] = cmplx_M_x(a_x[j], a_y[j], phi_x, phi_y)
        a_y[j] = cmplx_M_y(tmp_1, a_y[j], phi_x, phi_y)

def FF2(a0_x: float32, a0_y: float32, a1_x: float32, a1_y: float32) -> float32[4]:
    d0: float32[4] = 0.0

    d0[0] = cmplx_add_x(a0_x, a1_x)
    d0[1] = cmplx_add_y(a0_y, a1_y)
    d0[2] = cmplx_sub_x(a0_x, a1_x)
    d0[3] = cmplx_sub_y(a0_y, a1_y)

    return d0

# def FFT4(a0_x: float32, a0_y: float32, a1_x: float32, a1_y: float32, a2_x: float32, a2_y: float32, a3_x: float32, a3_y: float32):
#     exp_1_44_x: float32 = 0
#     exp_1_44_y: float32 = -1

#     FF2(a0_x, a0_y, a2_x, a2_y)

#     # FF2(a1_x, a1_y, a3_x, a3_y)
#     # tmp = a3_x

#     # a3_x = a3_x * exp_1_44_x - a3_y * exp_1_44_y
#     # a3_y = tmp * exp_1_44_y - a3_y * exp_1_44_x

#     # FF2(a0_x, a0_y, a1_x, a1_y)
#     # FF2(a2_x, a2_y, a3_x, a3_y)

def FFT4_1(a_x: float32[8], a_y: float32[8]):
    exp_1_44_x: float32 = 0.0
    exp_1_44_y: float32 = -1.0

    tmp_1 = FF2(a_x[0], a_y[0], a_x[2], a_y[2])
    a_x[0] = tmp_1[0]
    a_y[0] = tmp_1[1]
    a_x[2] = tmp_1[2]
    a_y[2] = tmp_1[3]

    tmp_2 = FF2(a_x[1], a_y[1], a_x[3], a_y[3])
    a_x[1] = tmp_2[0]
    a_y[1] = tmp_2[1]
    a_x[3] = tmp_2[2]
    a_y[3] = tmp_2[3]

    tmp_3: float32 = a_x[3]

    a_x[3] = a_x[3] * exp_1_44_x - a_y[3] * exp_1_44_y
    a_y[3] = tmp_3 * exp_1_44_y - a_y[3] * exp_1_44_x

    tmp_4 = FF2(a_x[0], a_y[0], a_x[1], a_y[1])
    a_x[0] = tmp_4[0]
    a_y[0] = tmp_4[1]
    a_x[1] = tmp_4[2]
    a_y[1] = tmp_4[3]

    tmp_5 = FF2(a_x[2], a_y[2], a_x[3], a_y[3])
    a_x[2] = tmp_5[0]
    a_y[2] = tmp_5[1]
    a_x[3] = tmp_5[2]
    a_y[3] = tmp_5[3]

    # FFT4(a_x[4], a_y[4], a_x[5], a_y[5], a_x[6], a_y[6], a_x[7], a_y[7])

def FFT4_2(a_x: float32[8], a_y: float32[8]):
    exp_1_44_x: float32 = 0.0
    exp_1_44_y: float32 = -1.0

    tmp_1 = FF2(a_x[4], a_y[4], a_x[6], a_y[6])
    a_x[4] = tmp_1[0]
    a_y[4] = tmp_1[1]
    a_x[6] = tmp_1[2]
    a_y[6] = tmp_1[3]

    tmp_2 = FF2(a_x[5], a_y[5], a_x[7], a_y[7])
    a_x[5] = tmp_2[0]
    a_y[5] = tmp_2[1]
    a_x[7] = tmp_2[2]
    a_y[7] = tmp_2[3]

    tmp_3: float32 = a_x[7]

    a_x[7] = a_x[7] * exp_1_44_x - a_y[7] * exp_1_44_y
    a_y[7] = tmp_3 * exp_1_44_y - a_y[7] * exp_1_44_x

    tmp_4 = FF2(a_x[4], a_y[4], a_x[5], a_y[5])
    a_x[4] = tmp_4[0]
    a_y[4] = tmp_4[1]
    a_x[5] = tmp_4[2]
    a_y[5] = tmp_4[3]

    tmp_5 = FF2(a_x[6], a_y[6], a_x[7], a_y[7])
    a_x[6] = tmp_5[0]
    a_y[6] = tmp_5[1]
    a_x[7] = tmp_5[2]
    a_y[7] = tmp_5[3]

def FFT8(a_x: float32[8], a_y: float32[8]):
    M_SQRT1_2: float32 = 0.70710678118654752440
    exp_1_8_x: float32 = 1.0
    exp_1_8_y: float32 = -1.0
    exp_1_4_x: float32 = 0.0
    exp_1_4_y: float32 = -1.0
    exp_3_8_x: float32 = -1.0
    exp_3_8_y: float32 = -1.0

    tmp_1 = FF2(a_x[0], a_y[0], a_x[4], a_y[4])
    a_x[0] = tmp_1[0]
    a_y[0] = tmp_1[1]
    a_x[4] = tmp_1[2]
    a_y[4] = tmp_1[3]

    tmp_2 = FF2(a_x[1], a_y[1], a_x[5], a_y[5])
    a_x[1] = tmp_2[0]
    a_y[1] = tmp_2[1]
    a_x[5] = tmp_2[2]
    a_y[5] = tmp_2[3]

    tmp_3 = FF2(a_x[2], a_y[2], a_x[6], a_y[6])
    a_x[2] = tmp_3[0]
    a_y[2] = tmp_3[1]
    a_x[6] = tmp_3[2]
    a_y[6] = tmp_3[3]

    tmp_4 = FF2(a_x[3], a_y[3], a_x[7], a_y[7])
    a_x[3] = tmp_4[0]
    a_y[3] = tmp_4[1]
    a_x[7] = tmp_4[2]
    a_y[7] = tmp_4[3]

    tmp_5: float32 = a_x[5]
    a_x[5] = cm_fl_mul_x(cmplx_mul_x(a_x[5], a_y[5], exp_1_8_x, exp_1_8_y), M_SQRT1_2)
    a_y[5] = cm_fl_mul_y(cmplx_mul_y(tmp_5, a_y[5], exp_1_8_x, exp_1_8_y), M_SQRT1_2)

    tmp_5 = a_x[6]
    a_x[6] = cmplx_mul_x(a_x[6], a_y[6], exp_1_4_x, exp_1_4_y)
    a_y[6] = cmplx_mul_y(tmp_5, a_y[6], exp_1_4_x, exp_1_4_y)

    tmp_5 = a_x[7]
    a_x[7] = cm_fl_mul_x(cmplx_mul_x(a_x[7], a_y[7], exp_3_8_x, exp_3_8_y), M_SQRT1_2)
    a_y[7] = cm_fl_mul_y(cmplx_mul_y(tmp_5, a_y[7], exp_3_8_x, exp_3_8_y), M_SQRT1_2)

    # FFT4(a_x[0], a_y[0], a_x[1], a_y[1], a_x[2], a_y[2], a_x[3], a_y[3])
    FFT4_1(a_x, a_y)
    # FFT4(a_x[4], a_y[4], a_x[5], a_y[5], a_x[6], a_y[6], a_x[7], a_y[7])
    FFT4_2(a_x, a_y)

def loadx8(a_x, x, offset, sx):
    a_x[0] = x[0 * sx + offset]
    a_x[1] = x[1 * sx + offset]
    a_x[2] = x[2 * sx + offset]
    a_x[3] = x[3 * sx + offset]
    a_x[4] = x[4 * sx + offset]
    a_x[5] = x[5 * sx + offset]
    a_x[6] = x[6 * sx + offset]
    a_x[7] = x[7 * sx + offset]

def loady8(a_y: float32[8], x: float32[8 * 8 * 9], offset: int32, sx: int32):
    a_y[0] = x[0 * sx + offset]
    a_y[1] = x[1 * sx + offset]
    a_y[2] = x[2 * sx + offset]
    a_y[3] = x[3 * sx + offset]
    a_y[4] = x[4 * sx + offset]
    a_y[5] = x[5 * sx + offset]
    a_y[6] = x[6 * sx + offset]
    a_y[7] = x[7 * sx + offset]

def fft1D_512(work_x: float32[512], work_y: float32[512]):
    stride: int32 = 64
    counter: int32 = 0
    reversed: int32[8] = [0, 4, 2, 6, 1, 5, 3, 7]

    DATA_x: float32[64 * 8] = 0.0
    DATA_y: float32[64 * 8] = 0.0

    data_x: float32[8] = 0.0
    data_y: float32[8] = 0.0

    smem: float32[8 * 8 * 9] = 0.0

    # # BUG: CANNOT REASSIGN ARRAY VALUES
    # for i in range(8):
    #     DATA_x[i] = 0.0
    #     data_y[i] = 0.0

    # Do it all at once...
    # Loop 1

    # BUG: WITH FOR-LOOP VARIABLE
    for tid in range(64):
        # GLOBAL_LOAD...
        data_x[0] = work_x[0 * stride + tid]
        data_x[1] = work_x[1 * stride + tid]
        data_x[2] = work_x[2 * stride + tid]
        data_x[3] = work_x[3 * stride + tid]
        data_x[4] = work_x[4 * stride + tid]
        data_x[5] = work_x[5 * stride + tid]
        data_x[6] = work_x[6 * stride + tid]
        data_x[7] = work_x[7 * stride + tid]

        data_y[0] = work_y[0 * stride + tid]
        data_y[1] = work_y[1 * stride + tid]
        data_y[2] = work_y[2 * stride + tid]
        data_y[3] = work_y[3 * stride + tid]
        data_y[4] = work_y[4 * stride + tid]
        data_y[5] = work_y[5 * stride + tid]
        data_y[6] = work_y[6 * stride + tid]
        data_y[7] = work_y[7 * stride + tid]

        # First 8 point FFT...
        FFT8(data_x, data_y)

        # First Twiddle
        twiddles8(data_x, data_y, counter, 512)

        # Save for fence
        DATA_x[tid * 8]     = data_x[0]
        DATA_x[tid * 8 + 1] = data_x[1]
        DATA_x[tid * 8 + 2] = data_x[2]
        DATA_x[tid * 8 + 3] = data_x[3]
        DATA_x[tid * 8 + 4] = data_x[4]
        DATA_x[tid * 8 + 5] = data_x[5]
        DATA_x[tid * 8 + 6] = data_x[6]
        DATA_x[tid * 8 + 7] = data_x[7]

        DATA_y[tid * 8]     = data_y[0]
        DATA_y[tid * 8 + 1] = data_y[1]
        DATA_y[tid * 8 + 2] = data_y[2]
        DATA_y[tid * 8 + 3] = data_y[3]
        DATA_y[tid * 8 + 4] = data_y[4]
        DATA_y[tid * 8 + 5] = data_y[5]
        DATA_y[tid * 8 + 6] = data_y[6]
        DATA_y[tid * 8 + 7] = data_y[7]

        counter += 1

    sx: int32 = 66
    # Loop 2
    for tid in range(64):
        tid_int: int32 = tid
        hi: index = tid_int >> 3
        lo: index = tid_int & 7
        offset: int32 = hi * 8 + lo
        
        smem[0 * sx + offset] = DATA_x[tid * 8 + 0]
        smem[4 * sx + offset] = DATA_x[tid * 8 + 1]
        smem[1 * sx + offset] = DATA_x[tid * 8 + 4]
        smem[5 * sx + offset] = DATA_x[tid * 8 + 5]
        smem[2 * sx + offset] = DATA_x[tid * 8 + 2]
        smem[6 * sx + offset] = DATA_x[tid * 8 + 3]
        smem[3 * sx + offset] = DATA_x[tid * 8 + 6]
        smem[7 * sx + offset] = DATA_x[tid * 8 + 7]

    sx = 8
    # Loop 3
    for tid in range(64):
        tid_int: int32 = tid
        hi: index = tid_int >> 3
        lo: index = tid_int & 7
        lo = tid & 7
        offset: int32 = lo * 66 + hi

        DATA_x[tid * 8 + 0] = smem[0 * sx + offset]
        DATA_x[tid * 8 + 4] = smem[4 * sx + offset]
        DATA_x[tid * 8 + 1] = smem[1 * sx + offset]
        DATA_x[tid * 8 + 5] = smem[5 * sx + offset]
        DATA_x[tid * 8 + 2] = smem[2 * sx + offset]
        DATA_x[tid * 8 + 6] = smem[6 * sx + offset]
        DATA_x[tid * 8 + 3] = smem[3 * sx + offset]
        DATA_x[tid * 8 + 7] = smem[7 * sx + offset]

    sx = 66
    # Loop 4
    for tid in range(64):
        tid_int: int32 = tid
        hi: index = tid_int >> 3
        lo: index = tid_int & 7
        offset: int32 = hi * 8 + lo

        smem[0 * sx + offset] = DATA_y[tid * 8 + 0]
        smem[4 * sx + offset] = DATA_y[tid * 8 + 1]
        smem[1 * sx + offset] = DATA_y[tid * 8 + 4]
        smem[5 * sx + offset] = DATA_y[tid * 8 + 5]
        smem[2 * sx + offset] = DATA_y[tid * 8 + 2]
        smem[6 * sx + offset] = DATA_y[tid * 8 + 3]
        smem[3 * sx + offset] = DATA_y[tid * 8 + 6]
        smem[7 * sx + offset] = DATA_y[tid * 8 + 7]

    # Loop 5
    for tid in range(64):
        data_y[0] = DATA_y[tid * 8 + 0]
        data_y[1] = DATA_y[tid * 8 + 1]
        data_y[2] = DATA_y[tid * 8 + 2]
        data_y[3] = DATA_y[tid * 8 + 3]
        data_y[4] = DATA_y[tid * 8 + 4]
        data_y[5] = DATA_y[tid * 8 + 5]
        data_y[6] = DATA_y[tid * 8 + 6]
        data_y[7] = DATA_y[tid * 8 + 7]

        tid_int: int32 = tid
        hi: index = tid_int >> 3
        lo: index = tid_int & 7

        # BUG: GET CASTING ERRORS WHEN INPUTTING TMP_1 RAW INTO LOADY8
        tmp_1: int32 = lo * 66 + hi

        loady8(data_y, smem, tmp_1, 8)

        DATA_y[tid * 8]     = data_y[0]
        DATA_y[tid * 8 + 1] = data_y[1]
        DATA_y[tid * 8 + 2] = data_y[2]
        DATA_y[tid * 8 + 3] = data_y[3]
        DATA_y[tid * 8 + 4] = data_y[4]
        DATA_y[tid * 8 + 5] = data_y[5]
        DATA_y[tid * 8 + 6] = data_y[6]
        DATA_y[tid * 8 + 7] = data_y[7]

    # Loop 6
    for tid in range(64):
        data_x[0] = DATA_x[tid * 8 + 0]
        data_x[1] = DATA_x[tid * 8 + 1]
        data_x[2] = DATA_x[tid * 8 + 2]
        data_x[3] = DATA_x[tid * 8 + 3]
        data_x[4] = DATA_x[tid * 8 + 4]
        data_x[5] = DATA_x[tid * 8 + 5]
        data_x[6] = DATA_x[tid * 8 + 6]
        data_x[7] = DATA_x[tid * 8 + 7]

        data_y[0] = DATA_y[tid * 8 + 0]
        data_y[1] = DATA_y[tid * 8 + 1]
        data_y[2] = DATA_y[tid * 8 + 2]
        data_y[3] = DATA_y[tid * 8 + 3]
        data_y[4] = DATA_y[tid * 8 + 4]
        data_y[5] = DATA_y[tid * 8 + 5]
        data_y[6] = DATA_y[tid * 8 + 6]
        data_y[7] = DATA_y[tid * 8 + 7]

        # Second FFT8...
        FFT8(data_x, data_y)

        # Calculate hi for second twiddle calculation...
        tid_int: int32 = tid
        hi: int32 = tid_int >> 3

        # Second twiddles calc, use hi and 64 stride version as defined in G80/SHOC...
        twiddles8(data_x, data_y, hi, 64)

        # Save for final transpose...
        DATA_x[tid * 8] = data_x[0]
        DATA_x[tid * 8 + 1] = data_x[1]
        DATA_x[tid * 8 + 2] = data_x[2]
        DATA_x[tid * 8 + 3] = data_x[3]
        DATA_x[tid * 8 + 4] = data_x[4]
        DATA_x[tid * 8 + 5] = data_x[5]
        DATA_x[tid * 8 + 6] = data_x[6]
        DATA_x[tid * 8 + 7] = data_x[7]

        DATA_y[tid * 8] = data_y[0]
        DATA_y[tid * 8 + 1] = data_y[1]
        DATA_y[tid * 8 + 2] = data_y[2]
        DATA_y[tid * 8 + 3] = data_y[3]
        DATA_y[tid * 8 + 4] = data_y[4]
        DATA_y[tid * 8 + 5] = data_y[5]
        DATA_y[tid * 8 + 6] = data_y[6]
        DATA_y[tid * 8 + 7] = data_y[7]

    # Transpose..
    sx = 72
    # Loop 7
    for tid in range(64):
        tid_int: int32 = tid
        hi: index = tid_int >> 3
        lo: index = tid_int & 7
        offset: int32 = hi * 8 + lo

        smem[0 * sx + offset] = DATA_x[tid * 8 + 0]
        smem[4 * sx + offset] = DATA_x[tid * 8 + 1]
        smem[1 * sx + offset] = DATA_x[tid * 8 + 4]
        smem[5 * sx + offset] = DATA_x[tid * 8 + 5]
        smem[2 * sx + offset] = DATA_x[tid * 8 + 2]
        smem[6 * sx + offset] = DATA_x[tid * 8 + 3]
        smem[3 * sx + offset] = DATA_x[tid * 8 + 6]
        smem[7 * sx + offset] = DATA_x[tid * 8 + 7]

    sx = 8
    # Loop 8
    for tid in range(64):
        tid_int: int32 = tid
        hi: index = tid_int >> 3
        lo: index = tid_int & 7
        offset: int32 = hi * 72 + lo

        DATA_x[tid * 8 + 0] = smem[0 * sx + offset]
        DATA_x[tid * 8 + 4] = smem[4 * sx + offset]
        DATA_x[tid * 8 + 1] = smem[1 * sx + offset]
        DATA_x[tid * 8 + 5] = smem[5 * sx + offset]
        DATA_x[tid * 8 + 2] = smem[2 * sx + offset]
        DATA_x[tid * 8 + 6] = smem[6 * sx + offset]
        DATA_x[tid * 8 + 3] = smem[3 * sx + offset]
        DATA_x[tid * 8 + 7] = smem[7 * sx + offset]

    sx = 72
    # Loop 9
    for tid in range(64):
        tid_int: int32 = tid
        hi: index = tid_int >> 3
        lo: index = tid_int & 7
        offset: int32 = hi * 8 + lo

        smem[0 * sx + offset] = DATA_y[tid * 8 + 0]
        smem[4 * sx + offset] = DATA_y[tid * 8 + 1]
        smem[1 * sx + offset] = DATA_y[tid * 8 + 4]
        smem[5 * sx + offset] = DATA_y[tid * 8 + 5]
        smem[2 * sx + offset] = DATA_y[tid * 8 + 2]
        smem[6 * sx + offset] = DATA_y[tid * 8 + 3]
        smem[3 * sx + offset] = DATA_y[tid * 8 + 6]
        smem[7 * sx + offset] = DATA_y[tid * 8 + 7]

    # Loop 10
    for tid in range(64):
        data_y[0] = DATA_y[tid * 8 + 0]
        data_y[1] = DATA_y[tid * 8 + 1]
        data_y[2] = DATA_y[tid * 8 + 2]
        data_y[3] = DATA_y[tid * 8 + 3]
        data_y[4] = DATA_y[tid * 8 + 4]
        data_y[5] = DATA_y[tid * 8 + 5]
        data_y[6] = DATA_y[tid * 8 + 6]
        data_y[7] = DATA_y[tid * 8 + 7]

        tid_int: int32 = tid
        hi: index = tid_int >> 3
        lo: index = tid_int & 7
        tmp_1: int32 = hi * 72 + lo

        loady8(data_y, smem, tmp_1, 8)

        DATA_y[tid * 8 + 0] = data_y[0]
        DATA_y[tid * 8 + 1] = data_y[1]
        DATA_y[tid * 8 + 2] = data_y[2]
        DATA_y[tid * 8 + 3] = data_y[3]
        DATA_y[tid * 8 + 4] = data_y[4]
        DATA_y[tid * 8 + 5] = data_y[5]
        DATA_y[tid * 8 + 6] = data_y[6]
        DATA_y[tid * 8 + 7] = data_y[7]

    # Loop 11
    for tid in range(64):
        # Load post-trans
        data_y[0] = DATA_y[tid * 8]
        data_y[1] = DATA_y[tid * 8 + 1]
        data_y[2] = DATA_y[tid * 8 + 2]
        data_y[3] = DATA_y[tid * 8 + 3]
        data_y[4] = DATA_y[tid * 8 + 4]
        data_y[5] = DATA_y[tid * 8 + 5]
        data_y[6] = DATA_y[tid * 8 + 6]
        data_y[7] = DATA_y[tid * 8 + 7]

        data_x[0] = DATA_x[tid * 8]
        data_x[1] = DATA_x[tid * 8 + 1]
        data_x[2] = DATA_x[tid * 8 + 2]
        data_x[3] = DATA_x[tid * 8 + 3]
        data_x[4] = DATA_x[tid * 8 + 4]
        data_x[5] = DATA_x[tid * 8 + 5]
        data_x[6] = DATA_x[tid * 8 + 6]
        data_x[7] = DATA_x[tid * 8 + 7]

        # Final 8pt FFT
        FFT8(data_x, data_y)

        # Global store
        work_x[0 * stride + tid] = data_x[reversed[0]]
        work_x[1 * stride + tid] = data_x[reversed[1]]
        work_x[2 * stride + tid] = data_x[reversed[2]]
        work_x[3 * stride + tid] = data_x[reversed[3]]
        work_x[4 * stride + tid] = data_x[reversed[4]]
        work_x[5 * stride + tid] = data_x[reversed[5]]
        work_x[6 * stride + tid] = data_x[reversed[6]]
        work_x[7 * stride + tid] = data_x[reversed[7]]

        work_y[0 * stride + tid] = data_y[reversed[0]]
        work_y[1 * stride + tid] = data_y[reversed[1]]
        work_y[2 * stride + tid] = data_y[reversed[2]]
        work_y[3 * stride + tid] = data_y[reversed[3]]
        work_y[4 * stride + tid] = data_y[reversed[4]]
        work_y[5 * stride + tid] = data_y[reversed[5]]
        work_y[6 * stride + tid] = data_y[reversed[6]]
        work_y[7 * stride + tid] = data_y[reversed[7]]

s = allo.customize(fft1D_512)

mod = s.build(target="llvm")

