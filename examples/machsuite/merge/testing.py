import os
import sys
import allo
import numpy as np
from allo.ir.types import int32

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
from mergesort import merge_sort

def read_data(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        data = [int(line.strip()) for line in lines if not line.startswith("%%")]
    return data

def main():
    input_data = read_data(os.path.join(_dir, "input.data"))
    check_data = read_data(os.path.join(_dir, "check.data"))

    values = np.array(input_data).astype(np.int32)
    
    check = np.array(check_data).astype(np.int32)


    s = allo.customize(merge_sort)

    mod = s.build()

    actual = mod(values)


    np.testing.assert_allclose(actual, check, rtol=1e-5, atol=1e-5)
    


if __name__ == "__main__":
    main()

