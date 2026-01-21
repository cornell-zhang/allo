import allo
import numpy as np
from mergesort import merge_sort
from allo.ir.types import int32

N = 5

def merge(a, start, m, stop):
    temp = np.zeros(N, dtype=np.int32)
    for i in range(start, m + 1):
        temp[i] = a[i]
    for j in range(m+1, stop + 1):
        temp[m + 1 + stop - j] = a[j]
    
    i = start
    j = stop
    for k in range(start, stop + 1):
        tmp_j = temp[j]
        tmp_i = temp[i]
        if (tmp_j < tmp_i):
            a[k] = tmp_j
            j -= 1
        else:
            a[k] = tmp_i
            i += 1

def mergesort_reference(a):
    start = 0
    stop = N - 1
    m = 1
    while m < stop-start + 1:
        for i in range(start, stop, m+m):
            f = i
            mid = i + m - 1
            to = i + m + m - 1
            if to < stop:
                merge(a, f, mid, to)
            else:
                merge(a, f, mid, stop)
        m += m
    return a


def read_data(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        data = [int(line.strip()) for line in lines if not line.startswith("%%")]
    return data

def main():
    # input_data = read_data("input1.data")
    # check_data = read_data("check1.data")

    # values = np.array(input_data).astype(np.int32)
    values = np.random.randint(0, 100, N).astype(np.int32)
    
    # check = np.array(check_data).astype(np.int32)

    check = values.copy()
    check.sort()

    #print(values)
    #print(check)

    # s = allo.customize(merge_sort)

    # mod = s.build()

    # actual = mod(values)
    # print(actual)

    mergesort_reference(values)

    np.testing.assert_allclose(values, check, rtol=1e-5, atol=1e-5)
    print("correct")


if __name__ == "__main__":
    main()
