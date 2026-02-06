# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import json
import allo
import numpy as np
from allo.ir.types import int32

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
import mergesort


def test_merge(psize="small"):
    setting_path = os.path.join(os.path.dirname(__file__), "..", "psize.json")
    with open(setting_path, "r") as fp:
        sizes = json.load(fp)
    params = sizes["merge"][psize]

    N = params["N"]
    mergesort.N = N

    np.random.seed(42)
    values = np.random.randint(0, 100000, size=N).astype(np.int32)

    s = allo.customize(mergesort.merge_sort)
    mod = s.build()

    actual = mod(values)
    expected = np.sort(values)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    print("PASS!")


if __name__ == "__main__":
    test_merge("full")
