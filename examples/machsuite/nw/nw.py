# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Needleman-Wunsch sequence alignment (MachSuite)

import allo
from allo.ir.types import int32

ALEN = 128
BLEN = 128
RESULT_LEN = ALEN + BLEN  # 256
MATRIX_SIZE = (ALEN + 1) * (BLEN + 1)  # 16641

# Scoring constants
MATCH_SCORE = 1
MISMATCH_SCORE = -1
GAP_SCORE = -1

# Direction constants for ptr matrix
ALIGN_VAL = 1
SKIPA_VAL = 2
SKIPB_VAL = 3


def needwun(SEQA: int32[ALEN], SEQB: int32[BLEN]) -> int32[2, RESULT_LEN]:
    M: int32[MATRIX_SIZE] = 0
    ptr: int32[MATRIX_SIZE] = 0
    result: int32[2, RESULT_LEN] = 0

    score: int32 = 0
    row_up: int32 = 0
    row: int32 = 0
    up_left: int32 = 0
    up: int32 = 0
    left: int32 = 0
    max_val: int32 = 0

    # Initialize first row: M[0, a] = a * GAP_SCORE
    for i in range(ALEN + 1):
        M[i] = i * GAP_SCORE

    # Initialize first column: M[b, 0] = b * GAP_SCORE
    for j in range(BLEN + 1):
        M[j * (ALEN + 1)] = j * GAP_SCORE

    # Fill DP matrix
    for bi in range(1, BLEN + 1):
        for ai in range(1, ALEN + 1):
            if SEQA[ai - 1] == SEQB[bi - 1]:
                score = MATCH_SCORE
            else:
                score = MISMATCH_SCORE

            row_up = (bi - 1) * (ALEN + 1)
            row = bi * (ALEN + 1)

            up_left = M[row_up + (ai - 1)] + score
            up = M[row_up + ai] + GAP_SCORE
            left = M[row + (ai - 1)] + GAP_SCORE

            # Compute max of three
            max_val = up_left
            if up > max_val:
                max_val = up
            if left > max_val:
                max_val = left

            M[row + ai] = max_val
            if max_val == left:
                ptr[row + ai] = SKIPB_VAL
            elif max_val == up:
                ptr[row + ai] = SKIPA_VAL
            else:
                ptr[row + ai] = ALIGN_VAL

    # Traceback
    a_idx: int32 = ALEN
    b_idx: int32 = BLEN
    a_str_idx: int32 = 0
    b_str_idx: int32 = 0
    r: int32 = 0

    for step in range(ALEN + BLEN):
        if a_idx > 0 or b_idx > 0:
            if a_idx == 0:
                # Must go up (SKIPA)
                result[0, a_str_idx] = 45
                result[1, b_str_idx] = SEQB[b_idx - 1]
                a_str_idx = a_str_idx + 1
                b_str_idx = b_str_idx + 1
                b_idx = b_idx - 1
            elif b_idx == 0:
                # Must go left (SKIPB)
                result[0, a_str_idx] = SEQA[a_idx - 1]
                result[1, b_str_idx] = 45
                a_str_idx = a_str_idx + 1
                b_str_idx = b_str_idx + 1
                a_idx = a_idx - 1
            else:
                r = b_idx * (ALEN + 1)
                if ptr[r + a_idx] == ALIGN_VAL:
                    result[0, a_str_idx] = SEQA[a_idx - 1]
                    result[1, b_str_idx] = SEQB[b_idx - 1]
                    a_str_idx = a_str_idx + 1
                    b_str_idx = b_str_idx + 1
                    a_idx = a_idx - 1
                    b_idx = b_idx - 1
                elif ptr[r + a_idx] == SKIPB_VAL:
                    result[0, a_str_idx] = SEQA[a_idx - 1]
                    result[1, b_str_idx] = 45
                    a_str_idx = a_str_idx + 1
                    b_str_idx = b_str_idx + 1
                    a_idx = a_idx - 1
                else:
                    result[0, a_str_idx] = 45
                    result[1, b_str_idx] = SEQB[b_idx - 1]
                    a_str_idx = a_str_idx + 1
                    b_str_idx = b_str_idx + 1
                    b_idx = b_idx - 1

    # Pad remaining positions with '_' (95)
    for idx in range(RESULT_LEN):
        if result[0, idx] == 0:
            result[0, idx] = 95
        if result[1, idx] == 0:
            result[1, idx] = 95

    return result


if __name__ == "__main__":
    s = allo.customize(needwun)
    print(s.module)
    mod = s.build()
    print("Build success!")
