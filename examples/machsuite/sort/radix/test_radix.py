# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import numpy as np
import allo
from radix_sort import ss_sort, SIZE

def test_radix_sort():
    data_dir = os.path.dirname(os.path.abspath(__file__))

    # Read input data
    input_path = os.path.join(data_dir, "input.data")
    values = []
    reading = False
    with open(input_path, 'r') as f:
        for line in f:
            if line.strip() == '%%':
                reading = True
                continue
            if reading:
                values.append(int(line.strip()))

    np_input = np.array(values[:SIZE], dtype=np.int32)

    # Read expected output
    check_path = os.path.join(data_dir, "check.data")
    expected = []
    reading = False
    with open(check_path, 'r') as f:
        for line in f:
            if line.strip() == '%%':
                reading = True
                continue
            if reading:
                expected.append(int(line.strip()))

    np_expected = np.array(expected[:SIZE], dtype=np.int32)

    # Build and run
    s = allo.customize(ss_sort)
    mod = s.build()
    result = mod(np_input)

    np.testing.assert_array_equal(result, np_expected)
    print("PASS!")

if __name__ == "__main__":
    test_radix_sort()
