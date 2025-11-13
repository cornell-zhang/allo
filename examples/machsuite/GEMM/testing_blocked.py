import allo
import numpy as np
from gemm_blocked import gemm
from allo.ir.types import int32

def read_data(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        data = [int(line.strip()) for line in lines if not line.startswith("%%")]
    return data

def main():
    input_data = read_data("input.data")

    m1,m2,n = np.array(input_data).astype(np.int32)


    s = allo.customize(gemm)

    mod = s.build(target="llvm")

    actual = mod(m1, m2, n) 
    check = np.matmul(m1, m2)
    np.testing.assert_allclose(actual, check, rtol=1e-5, atol=1e-5)
    


if __name__ == "__main__":
    main()