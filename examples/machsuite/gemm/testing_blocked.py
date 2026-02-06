import allo
import numpy as np
from gemm_blocked import bbgemm
from allo.ir.types import int32

def main():
    m1 = np.random.randint(0, 10, (1024, 1024)).astype(np.int32)
    m2 = np.random.randint(0, 10, (1024, 1024)).astype(np.int32)

    s = allo.customize(bbgemm)
    mod = s.build(target="llvm")

    actual = mod(m1, m2)
    check = np.matmul(m1.astype(np.int64), m2.astype(np.int64)).astype(np.int32)
    np.testing.assert_allclose(actual, check, rtol=1e-5, atol=1e-5)
    print("PASS!")


if __name__ == "__main__":
    main()
