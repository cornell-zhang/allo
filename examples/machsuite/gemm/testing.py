import os
import sys
import allo
import numpy as np
from allo.ir.types import float32

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
from gemm_ncubed import gemm


def main():
    m1 = np.random.randint(0, 100, (64, 64)).astype(np.float32)
    m2 = np.random.randint(0, 100, (64, 64)).astype(np.float32)
    # print(m1,m2)
    s = allo.customize(gemm)

    mod = s.build(target="llvm")

    actual = mod(m1, m2) 
    check = np.matmul(m1, m2)
    np.testing.assert_allclose(actual, check, rtol=1e-5, atol=1e-5)
    print("PASS!")


if __name__ == "__main__":
    main()