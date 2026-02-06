# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import numpy as np
import allo

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
from nw import needwun, ALEN, BLEN, RESULT_LEN


def nw_reference(seq_a, seq_b):
    """Python reference for Needleman-Wunsch alignment."""
    MATCH_SCORE = 1
    MISMATCH_SCORE = -1
    GAP_SCORE = -1
    ALIGN_VAL = 1
    SKIPA_VAL = 2
    SKIPB_VAL = 3

    alen = len(seq_a)
    blen = len(seq_b)
    result_len = alen + blen

    M = np.zeros((blen + 1, alen + 1), dtype=np.int32)
    ptr = np.zeros((blen + 1, alen + 1), dtype=np.int32)

    for i in range(alen + 1):
        M[0, i] = i * GAP_SCORE
    for j in range(blen + 1):
        M[j, 0] = j * GAP_SCORE

    for bi in range(1, blen + 1):
        for ai in range(1, alen + 1):
            score = MATCH_SCORE if seq_a[ai - 1] == seq_b[bi - 1] else MISMATCH_SCORE
            up_left = M[bi - 1, ai - 1] + score
            up = M[bi - 1, ai] + GAP_SCORE
            left = M[bi, ai - 1] + GAP_SCORE

            max_val = max(up_left, up, left)
            M[bi, ai] = max_val
            if max_val == left:
                ptr[bi, ai] = SKIPB_VAL
            elif max_val == up:
                ptr[bi, ai] = SKIPA_VAL
            else:
                ptr[bi, ai] = ALIGN_VAL

    result = np.zeros((2, result_len), dtype=np.int32)
    a_idx, b_idx = alen, blen
    idx = 0
    for _ in range(result_len):
        if a_idx > 0 or b_idx > 0:
            if a_idx == 0:
                result[0, idx] = 45  # '-'
                result[1, idx] = seq_b[b_idx - 1]
                idx += 1
                b_idx -= 1
            elif b_idx == 0:
                result[0, idx] = seq_a[a_idx - 1]
                result[1, idx] = 45  # '-'
                idx += 1
                a_idx -= 1
            else:
                if ptr[b_idx, a_idx] == ALIGN_VAL:
                    result[0, idx] = seq_a[a_idx - 1]
                    result[1, idx] = seq_b[b_idx - 1]
                    idx += 1
                    a_idx -= 1
                    b_idx -= 1
                elif ptr[b_idx, a_idx] == SKIPB_VAL:
                    result[0, idx] = seq_a[a_idx - 1]
                    result[1, idx] = 45  # '-'
                    idx += 1
                    a_idx -= 1
                else:
                    result[0, idx] = 45  # '-'
                    result[1, idx] = seq_b[b_idx - 1]
                    idx += 1
                    b_idx -= 1

    # Pad with '_' (95)
    for i in range(result_len):
        if result[0, i] == 0:
            result[0, i] = 95
        if result[1, i] == 0:
            result[1, i] = 95

    return result


def test_nw():
    np.random.seed(42)

    # Generate random DNA-like sequences
    alphabet = [ord(c) for c in "ACGT"]
    seq_a = np.array([alphabet[i] for i in np.random.randint(0, 4, size=ALEN)], dtype=np.int32)
    seq_b = np.array([alphabet[i] for i in np.random.randint(0, 4, size=BLEN)], dtype=np.int32)

    # Build and run
    s = allo.customize(needwun)
    mod = s.build()
    out = mod(seq_a, seq_b)

    # Python reference
    expected = nw_reference(seq_a, seq_b)

    np.testing.assert_array_equal(out, expected)
    print("PASS!")

if __name__ == "__main__":
    test_nw()
