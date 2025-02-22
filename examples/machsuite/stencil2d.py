import allo
import numpy as np
from allo.ir.types import int32

col_size = 64
row_size = 128
f_size = 9

def stencil2d(orig: int32[row_size, col_size], filter: int32[f_size] ) -> int32[row_size, col_size]:
    sol: int32[row_size, col_size] = 0
    for i, j in allo.grid(row_size-2, col_size-2):
        temp: int32= 0
        for m, n in allo.grid(3, 3):
            mul: int32= filter[m*3 + n] * orig[(i+m), (j+n)]
            temp += mul
        sol[i, j] = temp
    return sol

s = allo.customize(stencil2d)

print(s.module)

mod = s.build(target="llvm")

# np_orig = np.random.randint(0, 100, (row_size, col_size)).astype(np.int32)
# np_filter = np.random.randint(0, 100, (f_size)).astype(np.int32)

# np_sol = mod(np_orig, np_filter)

# test_sol = np.zeros((row_size - 2, col_size - 2))
# for r in range(row_size - 2):
#     for c in range(col_size - 2):
#         test_temp = 0
#         for k1 in range(3):
#             for k2 in range(3):
#                 test_mul = np_filter[k1, k2] * np_orig[r + k1, c + k2]
#                 test_temp += test_mul
#         test_sol[r, c] = test_temp

# np.testing.assert_allclose(np_sol, test_sol, rtol=1e-5, atol=1e-5)

import numpy as np

def read_data(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    start_index1 = content.find("%%")
    if start_index1 == -1:
        raise ValueError("Cannot find first '%%' in the file.")

    start_index2 = content.find("%%", start_index1 + 2)
    if start_index2 == -1:
        raise ValueError("Cannot find second '%%' in the file.")

    data_str1 = content[start_index1 + 2:start_index2].strip()
    data_list1 = [np.int32(line) for line in data_str1.split('\n')]

    np_orig = np.array(data_list1, dtype=np.int32).reshape(128, 64)

    data_str2 = content[start_index2 + 2:].strip()
    data_list2 = [np.int32(line) for line in data_str2.split('\n')]

    np_filter = np.array(data_list2, dtype=np.int32)

    return np_orig, np_filter

def read_check_data(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    start_index = content.find("%%")
    if start_index == -1:
        raise ValueError("Cannot find '%%' in the file.")

    data_str = content[start_index + 2:].strip()
    data_list = [np.int32(line) for line in data_str.split('\n')]

    check_sol = np.array(data_list, dtype=np.int32).reshape(128, 64)

    return check_sol

file_path_data = "input.data"
file_path_check = "check.data"
np_orig, np_filter = read_data(file_path_data)
check_sol = read_check_data(file_path_check)
# print("np_orig:")
# print(np_orig)
# print("\nnp_filter:")
# print(np_filter)
# print("\ncheck_sol:")
# print(check_sol)

np_sol = mod(np_orig, np_filter)
np.testing.assert_allclose(np_sol, check_sol, rtol=1e-5, atol=1e-5)