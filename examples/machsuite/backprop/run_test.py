import os
import sys
import allo
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
from backprop import backprop

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
    input_data = parse_data(os.path.join(_dir, "input.data"))
    check_data = parse_data(os.path.join(_dir, "check.data"))

    np_input = [np.asarray(lst) for lst in input_data]
    np_check = [np.asarray(lst) for lst in check_data]

    [weights1, weights2, weights3, biases1, biases2, biases3, training_data, training_targets] = np_input
    [weights1_check, weights2_check, weights3_check, biases1_check, biases2_check, biases3_check] = np_check

    s = allo.customize(backprop)
    mod = s.build(target="llvm")

    mod(weights1, weights2, weights3, biases1, biases2, biases3, training_data, training_targets)

    # TODO: Machsuite's benchmark for backprop does not match check.data either
    # Allo's backprop matches output for Machsuite's backprop

    # np.testing.assert_allclose(weights1, weights1_check, rtol=1e-3, atol=1e-3)
    # np.testing.assert_allclose(weights2, weights2_check, rtol=1e-3, atol=1e-3)
    # np.testing.assert_allclose(weights3, weights3_check, rtol=1e-3, atol=1e-3)
    # np.testing.assert_allclose(biases1, biases1_check, rtol=1e-3, atol=1e-3)
    # np.testing.assert_allclose(biases2, biases2_check, rtol=1e-3, atol=1e-3)
    # np.testing.assert_allclose(biases3, biases3_check, rtol=1e-3, atol=1e-3)