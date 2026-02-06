# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import numpy as np
import allo

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
from radix_sort import ss_sort, SIZE

def test_radix_sort():
    np.random.seed(42)
    np_input = np.random.randint(0, 100000, size=SIZE).astype(np.int32)

    s = allo.customize(ss_sort)
    mod = s.build()
    result = mod(np_input)

    expected = np.sort(np_input)
    np.testing.assert_array_equal(result, expected)
    print("PASS!")

if __name__ == "__main__":
    test_radix_sort()
