# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Radix Sort (MachSuite)

import allo
from allo.ir.types import int32

SIZE = 2048
NUMOFBLOCKS = 512
ELEMENTSPERBLOCK = 4
RADIXSIZE = 4
BUCKETSIZE = NUMOFBLOCKS * RADIXSIZE  # 2048
SCAN_BLOCK = 16
SCAN_RADIX = BUCKETSIZE // SCAN_BLOCK  # 128


def ss_sort(a: int32[SIZE]) -> int32[SIZE]:
    b: int32[SIZE] = 0
    bucket: int32[BUCKETSIZE] = 0
    sm: int32[SCAN_RADIX] = 0

    # Temporary variables
    bucket_indx: int32 = 0
    a_indx: int32 = 0
    valid_buffer: int32 = 0

    for exp in range(16):
        # init bucket
        for i_init in range(BUCKETSIZE):
            bucket[i_init] = 0

        # hist - build histogram
        if valid_buffer == 0:
            for blockID in range(NUMOFBLOCKS):
                for i_h in range(4):
                    a_indx = blockID * ELEMENTSPERBLOCK + i_h
                    bucket_indx = (
                        ((a[a_indx] >> (exp * 2)) & 0x3) * NUMOFBLOCKS + blockID + 1
                    )
                    bucket[bucket_indx] = bucket[bucket_indx] + 1
        else:
            for blockID in range(NUMOFBLOCKS):
                for i_h in range(4):
                    a_indx = blockID * ELEMENTSPERBLOCK + i_h
                    bucket_indx = (
                        ((b[a_indx] >> (exp * 2)) & 0x3) * NUMOFBLOCKS + blockID + 1
                    )
                    bucket[bucket_indx] = bucket[bucket_indx] + 1

        # local_scan
        for radixID in range(SCAN_RADIX):
            for i_ls in range(1, SCAN_BLOCK):
                bucket_indx = radixID * SCAN_BLOCK + i_ls
                bucket[bucket_indx] = bucket[bucket_indx] + bucket[bucket_indx - 1]

        # sum_scan
        sm[0] = 0
        for radixID_s in range(1, SCAN_RADIX):
            bucket_indx = radixID_s * SCAN_BLOCK - 1
            sm[radixID_s] = sm[radixID_s - 1] + bucket[bucket_indx]

        # last_step_scan
        for radixID_l in range(SCAN_RADIX):
            for i_lss in range(SCAN_BLOCK):
                bucket_indx = radixID_l * SCAN_BLOCK + i_lss
                bucket[bucket_indx] = bucket[bucket_indx] + sm[radixID_l]

        # update
        if valid_buffer == 0:
            for blockID_u in range(NUMOFBLOCKS):
                for i_u in range(4):
                    bucket_indx = (
                        (a[blockID_u * ELEMENTSPERBLOCK + i_u] >> (exp * 2)) & 0x3
                    ) * NUMOFBLOCKS + blockID_u
                    a_indx = blockID_u * ELEMENTSPERBLOCK + i_u
                    b[bucket[bucket_indx]] = a[a_indx]
                    bucket[bucket_indx] = bucket[bucket_indx] + 1
            valid_buffer = 1
        else:
            for blockID_u in range(NUMOFBLOCKS):
                for i_u in range(4):
                    bucket_indx = (
                        (b[blockID_u * ELEMENTSPERBLOCK + i_u] >> (exp * 2)) & 0x3
                    ) * NUMOFBLOCKS + blockID_u
                    a_indx = blockID_u * ELEMENTSPERBLOCK + i_u
                    a[bucket[bucket_indx]] = b[a_indx]
                    bucket[bucket_indx] = bucket[bucket_indx] + 1
            valid_buffer = 0

    # After 16 passes (even count), result is in buffer A
    return a


if __name__ == "__main__":
    s = allo.customize(ss_sort)
    print(s.module)
    mod = s.build()
    print("Build success!")
