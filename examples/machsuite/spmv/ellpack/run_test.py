import os
import sys
import allo
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
from ellpack import ellpack

def parse_data(file):
    data_arrays = []
    current_array = []

    with open(file, 'r') as f:
        for line in f:
            if line.strip() == '%%':
                if current_array:
                    data_arrays.append(current_array)
                    current_array = []
            else:
                num = float(line.strip())
                current_array.append(num)

    data_arrays.append(current_array)
    return data_arrays

if __name__ == "__main__":
    input = parse_data(os.path.join(_dir, "input.data"))
    check = parse_data(os.path.join(_dir, "check.data"))

    values = np.array(input[0]).astype(np.float64)
    columns = np.array(input[1]).astype(np.int32)
    vector = np.array(input[2]).astype(np.float64)

    check = np.array(check[0]).astype(np.float64)

    ##Building
    s = allo.customize(ellpack)
    mod = s.build()

    observed = mod(values, columns, vector)
    np.testing.assert_allclose(observed, check, rtol=1e-5, atol=1e-5)