import os
import sys
import allo
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
from aes import encrypt_ecb


def test_aes():
    # AES-256 test vector: key = 0..31, plaintext = 0x00,0x11,...,0xFF
    k = np.array(list(range(32)), dtype=np.uint8)
    buf = np.array([0, 17, 34, 51, 68, 85, 102, 119,
                    136, 153, 170, 187, 204, 221, 238, 255], dtype=np.uint8)

    # Known expected ciphertext from MachSuite reference
    expected = np.array([142, 162, 183, 202, 81, 103, 69, 191,
                         234, 252, 73, 144, 75, 73, 96, 137], dtype=np.uint8)

    s = allo.customize(encrypt_ecb)
    mod = s.build(target="llvm")
    mod(k, buf)

    np.testing.assert_array_equal(buf, expected)
    print("PASS!")


if __name__ == "__main__":
    test_aes()
