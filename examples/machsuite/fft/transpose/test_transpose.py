# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# test_transpose.py

import os
import numpy as np
import allo

from transpose_fft import fft1D_512, mod

from allo.ir.types import int32, float32, float64

def test_transpose_fft():
    data_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(data_dir, "input_transpose.data")
    counter = 0
    real = []
    img = []

    with open(file_path, 'r') as file:
        for line in file:
            number = line.strip()
            if "%%" in line:
                counter += 1
                continue
            if counter == 1:
                real.append(float(number))
            elif counter == 2:
                img.append(float(number))

    real = np.array(real, dtype=np.float32)
    img = np.array(img, dtype=np.float32)

    mod(real, img)

    output_path = os.path.join(data_dir, "check_transpose.data")
    counter = 0

    golden_real = []
    golden_img = []
    with open(output_path, 'r') as file:
        for line in file:
            number = line.strip()
            if "%%" in line:
                counter += 1
                continue
            if counter == 1:
                golden_real.append(float(number))
            elif counter == 2:
                golden_img.append(float(number))

    np.testing.assert_allclose(real, golden_real, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(img, golden_img, rtol=1e-5, atol=1e-5)
    print("PASS!")

test_transpose_fft()
