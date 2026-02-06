import allo
from allo.ir.types import int32
import numpy as np
N = 2048


def merge(a:int32[N], start: int32, m: int32, stop: int32):
    temp:int32[N]
   
    tmp_j:int32
    tmp_i:int32

    i:int32 = start
    j:int32 = stop

    for index in range(start, m + 1):
        temp[index] = a[index]
    
    for index in range(m+1, stop + 1):
        temp[m + 1 + stop - index] = a[index]

    for k in range(start, stop + 1):
        tmp_j = temp[j]
        tmp_i = temp[i]

        if (tmp_j < tmp_i):
            a[k] = tmp_j
            j -= 1
        else:
            a[k] = tmp_i
            i += 1


def merge_sort(a:int32[N]) -> int32[N]:
    start:int32 = 0
    stop:int32 = N - 1

    f:int32
    m:int32 = 1
    mid:int32
    to:int32


    while (m < stop-start + 1):
        for ii in range(start, stop, m+m):
            f = ii

            mid = ii + m - 1

            to = ii + m + m - 1
            if (to <= stop):
                merge(a, f, mid, to)
            else:
                merge(a, f, mid, stop)

        m += m
       

    return a
    
if __name__ == "__main__":
    s = allo.customize(merge_sort)
    mod = s.build()

    # Test
    a = np.random.randint(0, 10000, N).astype(np.int32)
    result = mod(a)
    expected = np.sort(a).astype(np.int32)
    np.testing.assert_array_equal(result, expected)
    print("PASS!")