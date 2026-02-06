# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import numpy as np
import allo
from nw import needwun, ALEN, BLEN, RESULT_LEN

def test_nw():
    data_dir = os.path.dirname(os.path.abspath(__file__))

    # Read input sequences
    input_path = os.path.join(data_dir, "input.data")
    sections = []
    with open(input_path, 'r') as f:
        current = []
        for line in f:
            if line.strip() == '%%':
                if current:
                    sections.append(''.join(current))
                current = []
            else:
                current.append(line.strip())
        if current:
            sections.append(''.join(current))

    seq_a_str = sections[0][:ALEN]
    seq_b_str = sections[1][:BLEN]

    # Convert to int32 arrays (ASCII values)
    seq_a = np.array([ord(c) for c in seq_a_str], dtype=np.int32)
    seq_b = np.array([ord(c) for c in seq_b_str], dtype=np.int32)

    # Build and run
    s = allo.customize(needwun)
    mod = s.build()
    out = mod(seq_a, seq_b)

    # Convert result back to strings
    aligned_a = ''.join(chr(c) for c in out[0])
    aligned_b = ''.join(chr(c) for c in out[1])

    # Read expected output
    check_path = os.path.join(data_dir, "check.data")
    check_sections = []
    with open(check_path, 'r') as f:
        current = []
        for line in f:
            if line.strip() == '%%':
                if current:
                    check_sections.append(''.join(current))
                current = []
            else:
                current.append(line.strip())
        if current:
            check_sections.append(''.join(current))

    expected_a = check_sections[0][:RESULT_LEN]
    expected_b = check_sections[1][:RESULT_LEN]

    assert aligned_a == expected_a, f"AlignedA mismatch!\nGot:      {aligned_a[:80]}...\nExpected: {expected_a[:80]}..."
    assert aligned_b == expected_b, f"AlignedB mismatch!\nGot:      {aligned_b[:80]}...\nExpected: {expected_b[:80]}..."
    print("PASS!")

if __name__ == "__main__":
    test_nw()
