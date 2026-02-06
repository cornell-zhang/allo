# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# test_strided.py

import os
import numpy as np
import allo

from strided_fft import fft, mod

from allo.ir.types import int32, float32, float64

def test_strided_fft():
    data_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(data_dir, "input_strided.data")
    counter = 0
    real = []
    img = []
    real_twid = []
    img_twid = []

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
            elif counter == 3:
                real_twid.append(float(number))
            elif counter == 4:
                img_twid.append(float(number))

    real = np.array(real, dtype=np.float32)
    img = np.array(img, dtype=np.float32)
    real_twid = np.array(real_twid, dtype=np.float32)
    img_twid = np.array(img_twid, dtype=np.float32)

    mod(real, img, real_twid, img_twid)

    output_path = os.path.join(data_dir, "check_strided.data")
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

test_strided_fft()
