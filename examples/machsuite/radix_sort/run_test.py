# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import json
import numpy as np
import allo

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
import radix_sort


def test_radix_sort(psize="small"):
    setting_path = os.path.join(os.path.dirname(__file__), "..", "psize.json")
    with open(setting_path, "r") as fp:
        sizes = json.load(fp)
    params = sizes["radix"][psize]

    SIZE = params["SIZE"]

    # Patch module constants and derived values
    radix_sort.SIZE = SIZE
    radix_sort.NUMOFBLOCKS = SIZE // 4
    radix_sort.ELEMENTSPERBLOCK = 4
    radix_sort.RADIXSIZE = 4
    radix_sort.BUCKETSIZE = radix_sort.NUMOFBLOCKS * radix_sort.RADIXSIZE
    radix_sort.SCAN_BLOCK = 16
    radix_sort.SCAN_RADIX = radix_sort.BUCKETSIZE // radix_sort.SCAN_BLOCK

    np.random.seed(42)
    np_input = np.random.randint(0, 100000, size=SIZE).astype(np.int32)

    s = allo.customize(radix_sort.ss_sort)
    mod = s.build()
    result = mod(np_input)

    expected = np.sort(np_input)
    np.testing.assert_array_equal(result, expected)
    print("PASS!")


if __name__ == "__main__":
    test_radix_sort("full")
