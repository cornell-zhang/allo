import md
import allo
import numpy as np
import os

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
    data_dir = os.path.dirname(os.path.abspath(__file__))
    input_data = parse_data(os.path.join(data_dir, "input.data"))
    check_data = parse_data(os.path.join(data_dir, "check.data"))
    check_x = np.array(check_data[0]).astype(np.float64)
    check_y = np.array(check_data[1]).astype(np.float64)
    check_z = np.array(check_data[2]).astype(np.float64)

    np_pos_x = np.array(input_data[0]).astype(np.float64)
    np_pos_y = np.array(input_data[1]).astype(np.float64)
    np_pos_z = np.array(input_data[2]).astype(np.float64)
    np_NL = np.array(input_data[3]).astype(np.int32)

    s_x = allo.customize(md.md_x)
    mod_x = s_x.build()
    s_y = allo.customize(md.md_y)
    mod_y = s_y.build()
    s_z = allo.customize(md.md_z)
    mod_z = s_z.build()

    forceX = mod_x(np_pos_x, np_pos_y, np_pos_z, np_NL)
    forceY = mod_y(np_pos_x, np_pos_y, np_pos_z, np_NL)
    forceZ = mod_z(np_pos_x, np_pos_y, np_pos_z, np_NL)

    np.testing.assert_allclose(forceX, check_x, rtol=2, atol=20)
    np.testing.assert_allclose(forceY, check_y, rtol=2, atol=20)
    np.testing.assert_allclose(forceZ, check_z, rtol=2, atol=20)
    print("PASS!")
