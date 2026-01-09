# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# test_transpose.py>

import pytest
import numpy as np
import allo

from transpose_fft import *

from allo.ir.types import int32, float32, float64

def test_transpose_fft():
    file_path = "/home/wty5/shared/allo/fft/transpose/input_transpose.data"
    counter = 0
    real = []
    img = []
    
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
    
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    
    fft1D_512(real, img)

    output_path = "/home/wty5/shared/allo/fft/transpose/check_transpose.data"
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
    
test_transpose_fft()