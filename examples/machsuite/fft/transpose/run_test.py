# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import numpy as np
import allo

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
from transpose_fft import fft1D_512


def test_transpose_fft(psize="small"):
    # fft_transpose is hardcoded to 512, no size parameters to patch
    s = allo.customize(fft1D_512)
    mod = s.build(target="llvm")

    np.random.seed(42)

    # Generate random complex input
    real = np.random.randn(512).astype(np.float32)
    img = np.random.randn(512).astype(np.float32)

    # Compute NumPy FFT reference from original input
    complex_input = real.astype(np.complex64) + 1j * img.astype(np.complex64)
    fft_ref = np.fft.fft(complex_input)

    # Run Allo kernel (in-place)
    mod(real, img)

    np.testing.assert_allclose(real, fft_ref.real, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(img, fft_ref.imag, rtol=1e-3, atol=1e-3)
    print("PASS!")


if __name__ == "__main__":
    test_transpose_fft("full")
