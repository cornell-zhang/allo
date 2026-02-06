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
    # C code uses dvector_t structs (AoS layout: x0,y0,z0,x1,y1,z1,...)
    force_data = np.array(check_data[0]).astype(np.float64)
    check_x = np.ascontiguousarray(force_data[0::3].reshape((4,4,4,10)))
    check_y = np.ascontiguousarray(force_data[1::3].reshape((4,4,4,10)))
    check_z = np.ascontiguousarray(force_data[2::3].reshape((4,4,4,10)))

    np_n_points = np.array(input_data[0]).astype(np.int32).reshape((4,4,4))
    pos_data = np.array(input_data[1]).astype(np.float64)
    np_pos_x = np.ascontiguousarray(pos_data[0::3].reshape((4,4,4,10)))
    np_pos_y = np.ascontiguousarray(pos_data[1::3].reshape((4,4,4,10)))
    np_pos_z = np.ascontiguousarray(pos_data[2::3].reshape((4,4,4,10)))

    s_x = allo.customize(md.md_x)
    mod_x = s_x.build()
    s_y = allo.customize(md.md_y)
    mod_y = s_y.build()
    s_z = allo.customize(md.md_z)
    mod_z = s_z.build()

    forceX = mod_x(np_n_points, np_pos_x, np_pos_y, np_pos_z)
    forceY = mod_y(np_n_points, np_pos_x, np_pos_y, np_pos_z)
    forceZ = mod_z(np_n_points, np_pos_x, np_pos_y, np_pos_z)

    np.testing.assert_allclose(forceX, check_x, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(forceY, check_y, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(forceZ, check_z, rtol=1e-5, atol=1e-5)
    print("PASS!")
