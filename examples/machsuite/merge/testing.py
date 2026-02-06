import os
import sys
import allo
import numpy as np
from allo.ir.types import int32

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
from mergesort import merge_sort


def main():
    np.random.seed(42)
    values = np.random.randint(0, 100000, size=2048).astype(np.int32)

    s = allo.customize(merge_sort)
    mod = s.build()

    actual = mod(values)
    expected = np.sort(values)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    print("PASS!")


if __name__ == "__main__":
    main()
