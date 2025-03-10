import allo
import numpy as np
from aes import encrypt_ecb

def parse_data(file_name):
    data_arrays = []
    current_array = []

    with open(file_name, 'r') as f:
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
    input_data = parse_data("input.data")
    check_data = parse_data("check.data")

    np_input = [np.asarray(lst).astype(np.uint8) for lst in input_data]
    np_check = [np.asarray(lst) for lst in check_data]

    [k, buf] = np_input
    [buf_check] = np_check

    s = allo.customize(encrypt_ecb)
    mod = s.build(target="llvm")
    mod(k, buf)

    np.testing.assert_allclose(buf, buf_check, rtol=1e-5, atol=1e-5)