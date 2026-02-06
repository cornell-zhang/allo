# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import json
import numpy as np
import math
import allo

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
import strided_fft


def python_strided_fft(real, img, real_twid, img_twid):
    """Python reference matching the Allo strided FFT (decimation-in-frequency)."""
    N = len(real)
    r = real.copy()
    im = img.copy()

    span = N >> 1
    log = 0
    while span > 0:
        odd = span
        while odd < N:
            odd |= span
            even = odd ^ span

            temp = r[even] + r[odd]
            r[odd] = r[even] - r[odd]
            r[even] = temp

            temp = im[even] + im[odd]
            im[odd] = im[even] - im[odd]
            im[even] = temp

            rootindex = (even << log) & (N - 1)
            if rootindex > 0:
                temp = real_twid[rootindex] * r[odd] - img_twid[rootindex] * im[odd]
                im[odd] = real_twid[rootindex] * im[odd] + img_twid[rootindex] * r[odd]
                r[odd] = temp

            odd += 1
        span >>= 1
        log += 1

    return r, im


def test_strided_fft(psize="small"):
    setting_path = os.path.join(os.path.dirname(__file__), "..", "..", "psize.json")
    with open(setting_path, "r") as fp:
        sizes = json.load(fp)
    params = sizes["fft_strided"][psize]

    FFT_SIZE = params["FFT_SIZE"]
    FFT_SIZE_HALF = FFT_SIZE // 2

    # Patch module constants
    strided_fft.FFT_SIZE = FFT_SIZE
    strided_fft.FFT_SIZE_HALF = FFT_SIZE_HALF

    s = allo.customize(strided_fft.fft)
    mod = s.build(target="llvm")

    np.random.seed(42)

    # Generate random complex input
    real = np.random.randn(FFT_SIZE).astype(np.float32)
    img = np.random.randn(FFT_SIZE).astype(np.float32)

    # Precompute twiddle factors
    real_twid = np.zeros(FFT_SIZE_HALF, dtype=np.float32)
    img_twid = np.zeros(FFT_SIZE_HALF, dtype=np.float32)
    for i in range(FFT_SIZE_HALF):
        angle = 2.0 * math.pi * i / FFT_SIZE
        real_twid[i] = np.float32(math.cos(angle))
        img_twid[i] = np.float32(math.sin(angle))

    # Compute Python reference
    ref_real, ref_img = python_strided_fft(real.copy(), img.copy(), real_twid, img_twid)

    # Run Allo kernel (in-place)
    mod(real, img, real_twid, img_twid)

    np.testing.assert_allclose(real, ref_real, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(img, ref_img, rtol=1e-5, atol=1e-5)
    print("PASS!")


if __name__ == "__main__":
    test_strided_fft("full")
