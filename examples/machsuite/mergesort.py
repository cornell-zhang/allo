
"""
import allo
from allo.ir.types import int32
import numpy as np


N = 10 #testing purpose
import numpy as np

def merge_sort(a:int32[N]):
    n:int32 = N
    size:int32 = 1

    while size < n - 1:
        
        left:int32 = 0

        while left < n - 1:
            mid:int32 = left + size - 1
            if(mid>n-1):
                mid = n-1
           
            right:int32 = left + 2 * size - 1
            if right>n-1:
                right= n-1

            if right >= n:
                right = n - 1

            merge(a, left, mid, right) #something wrong with the merge function

            left = left + size*2

        size = size*2

def merge(a:int32[N], start: int32, mid: int32, stop: int32):
    #temp:int32 = np.zeros(stop - start + 1, dtype=np.int32) #something wrong with this line
    temp:int32[N] = 0 #changed to this but getting a runtime error saying unsupported node "List"

    i:int32 = start
    j:int32 = mid + 1
    k:int32 = 0


    while i <= mid or j <= stop:
        if j > stop or (i <= mid and a[i] <= a[j]):
            temp[k] = a[i]
            i += 1
        else:
            temp[k] = a[j]
            j += 1
        k += 1

    #a[start:stop + 1] = temp[:k]

    for i in range(k):
        a[start + i] = temp[i]

test_array = np.array([1,2,3,4,5,9,8,7,6,0], dtype=np.int32)
merge_sort(test_array)
print("Sorted :", test_array)
s = allo.customize(merge_sort)
mod = s.build()
print(s.module)


"""

import allo
from allo.ir.types import int32
import numpy as np


N = 10 #testing purpose

def merge_sort(a:int32[N]):
    n:int32 = N
    size:int32 = 1

    while size < n - 1:
        
        left:int32 = 0

        while left < n - 1:
            mid:int32 = left + size - 1
            if(mid>n-1):
                mid = n-1
           
            right:int32 = left + 2 * size - 1
            if right>n-1:
                right= n-1

            if right >= n:
                right = n - 1

            merge(a, left, mid, right)

            left = left + size*2

        size = size*2

def merge(a:int32[N], start: int32, mid: int32, stop: int32):
    # temp:int32 = np.zeros(stop - start + 1, dtype=np.int32)
    temp: int32[N] = 0

    i:int32 = start
    j:int32 = mid + 1
    k:int32 = 0


    while i <= mid or j <= stop:
        if j > stop or (i <= mid and a[i] <= a[j]):
            temp[k] = a[i]
            i += 1
        else:
            temp[k] = a[j]
            j += 1
        k += 1

    # a[start:stop + 1] = temp[:k]
    for i in range(k):
        a[start+i] = temp[i]


s = allo.customize(merge_sort)
mod = s.build()
print(s.module)