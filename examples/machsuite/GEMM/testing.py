import allo
import numpy as np
from gemm_ncubed import gemm
from allo.ir.types import float32

def read_data(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        data = [float(line.strip()) for line in lines if not line.startswith("%%")]
    return data


def main():
    input_data = read_data("input.data")

    # m1= np.array(input_data[0:1024]).astype(np.float32).reshape((1024,1024))
    
    # m2= np.array(input_data[40:80]).astype(np.float32).reshape((1024,1024))
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