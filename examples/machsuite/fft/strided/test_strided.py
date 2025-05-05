# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# test_strided.py>

import pytest
import numpy as np
import allo

from strided_fft import fft

from allo.ir.types import int32, float32, float64

def test_strided_fft():
    file_path = "/home/wty5/shared/allo/fft/strided/input_strided.data"
    counter = 0
    real = []
    img = []
    real_twid = []
    img_twid = []
    
    try:
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
    
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    
    fft(real, img, real_twid, img_twid)

    output_path = "/home/wty5/shared/allo/fft/strided/check_strided.data"
    counter = 0

    golden_real = []
    golden_img = []
    try:
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
    
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")

    np.testing.assert_allclose(real, golden_real, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(img, golden_img, rtol=1e-5, atol=1e-5)
    
test_strided_fft()