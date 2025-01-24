# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# strided_fft.py

import allo
from allo.ir.types import float32, int32

FFT_SIZE = 1024
FFT_SIZE_HALF = int(FFT_SIZE / 2)

# void fft(double real[FFT_SIZE], double img[FFT_SIZE], double real_twid[FFT_SIZE/2], double img_twid[FFT_SIZE/2]){
def fft(real: float32[FFT_SIZE], img: float32[FFT_SIZE], real_twid: float32[FFT_SIZE_HALF], img_twid: float32[FFT_SIZE_HALF]):
    span: int32 = FFT_SIZE >> 1
    log:  int32 = 0
    even: int32 = 0
    odd:  int32 = 0
    rootindex: int32 = 0
    temp: float32 = 0.0
    
    # outer loop iterates over different stages of FFT
    while (span > 0):
        odd = span
        while (odd < FFT_SIZE):
            # odd index, with arr[0] = 1st index, so even is odd

            odd |= span

            # even index, with arr[1] = 2nd index, so odd is even
            even = odd ^ span

            # butterfly algorithm, temp stores intermediate results
            temp = real[even] + real[odd]

            # output[odd] = input[even] - input[odd]
            real[odd] = real[even] - real[odd]

            # output[even] = input[even] + input[odd]
            real[even] = temp

            temp = img[even] + img[odd]

            # output[odd] = input[even] + input[odd]
            img[odd] = img[even] - img[odd]

            # output[even] = input[even] + input[odd]
            img[even] = temp

            # finds index of precomputed twiddle factor

            rootindex = (even << log) & (FFT_SIZE - 1)

            # if twiddle factor is non-zero
            if rootindex > 0: 
                # real part is updated
                temp = real_twid[rootindex] * real[odd] - img_twid[rootindex] * img[odd]

                # imaginary part is updated
                img[odd] = real_twid[rootindex] * img[odd] + img_twid[rootindex] * real[odd]
                real[odd] = temp
            odd += 1
        # keeps track of stages using log
        span >>= 1
        log += 1

s = allo.customize(fft)

mod = s.build(target="llvm")