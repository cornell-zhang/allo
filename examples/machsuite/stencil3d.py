import allo
from allo.ir.types import int32

height_size = 32
col_size = 32
row_size = 16

def stencil3d(C: int32[2], orig: int32[row_size, col_size, height_size]) -> int32[row_size, col_size, height_size]:
    sol: int32[row_size, col_size, height_size] = 0
    sum0: int32 = 0
    sum1: int32 = 0
    mul0: int32 = 0
    mul1: int32 = 0

    # Handle boundary conditions by filling with original values     
    for j, k in allo.grid(col_size, row_size):
        sol[k, j, 0] = orig[k, j, 0]
        sol[k, j, height_size - 1] = orig[k, j, height_size - 1]

    for i, k in allo.grid(height_size - 1, row_size):
        sol[k, 0, (i+1)] = orig[k, 0, (i+1)]
        sol[k, col_size - 1, (i+1)] = orig[k, col_size - 1, (i+1)]

    for j, i in allo.grid(col_size-2, height_size-2):
        sol[0, (j+1), (i+1)] = orig[0, (j+1), (i+1)]
        sol[row_size - 1, (j+1), (i+1)] = orig[row_size - 1, (j+1), (i+1)]
   
    # Stencil computation
    for i, j, k in allo.grid( height_size - 2, col_size - 2, row_size - 2 ):
        sum0 = orig[(k+1), (j+1), (i+1)]
        sum1 = (orig[(k+1), (j+1), (i+2)] +
                orig[(k+1), (j+1), i] +
                orig[(k+1), (j+2), (i+1)] +
                orig[(k+1), j, (i+1)] +
                orig[(k+2), (j+1), (i+1)] +
                orig[k, (j+1), (i+1)])
        mul0 = sum0 * C[0]
        mul1 = sum1 * C[1]
        sol[(k+1), (j+1), (i+1)] = mul0 + mul1    

    return sol

s = allo.customize(stencil3d)

print(s.module)

mod = s.build(target="llvm")

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

    np_C = np.array(data_list1, dtype=np.int32)

    data_str2 = content[start_index2 + 2:].strip()
    data_list2 = [np.int32(line) for line in data_str2.split('\n')]

    #np_orig = np.array(data_list2, dtype=np.int32).reshape(row_size, col_size, height_size)

    np_orig = np.zeros((row_size, col_size, height_size), dtype=np.int32)

    index = 0
    for i in range(height_size):
        for j in range(col_size):
            for k in range(row_size):
                np_orig[k, j, i] = data_list2[index]
                index += 1

    return np_orig, np_C

def read_check_data(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    start_index = content.find("%%")
    if start_index == -1:
        raise ValueError("Cannot find '%%' in the file.")

    data_str = content[start_index + 2:].strip()
    data_list = [np.int32(line) for line in data_str.split('\n')]

    #check_sol = np.array(data_list, dtype=np.int32).reshape(row_size, col_size, height_size)

    check_sol = np.zeros((row_size, col_size, height_size), dtype=np.int32)

    index = 0
    for i in range(height_size):
        for j in range(col_size):
            for k in range(row_size):
                check_sol[k, j, i] = data_list[index]
                index += 1

    return check_sol

file_path_data = "input.data"
file_path_check = "check.data"
np_orig, np_C = read_data(file_path_data)
check_sol = read_check_data(file_path_check)
print("\nnp_C:")
print(np_C)
print("np_orig:")
print(np_orig)
# print("\ncheck_sol:")
# print(check_sol)

np_sol = mod(np_C, np_orig)
print("\ncheck_sol:")
print(check_sol)
print("\nnp_sol:")
print(np_sol)

with open("output.txt", "w") as file:
    for i in range(16):
        for j in range(32):
            for k in range(32):
                file.write(str(np_sol[i][j][k]) + "\n")

np.testing.assert_allclose(np_sol, check_sol, rtol=1e-5, atol=1e-5, verbose=True)
# result = np.array_equal(check_sol, np_sol)
# print("equal", result) 
# arrays_are_equal = np.array_equal(check_sol, np_sol)

# if arrays_are_equal:
#     print("equal")
# else:
#     print("not equal")
#     unequal_indices = np.where(check_sol != np_sol)
#     print("not equal position:")
#     print(list(zip(*unequal_indices)))
#     num_unequal_elements = len(unequal_indices[0])
#     print("not equal num:", num_unequal_elements)