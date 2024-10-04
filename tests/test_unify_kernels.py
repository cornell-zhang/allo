import allo
from allo.ir.transform import unify_kernels
from allo.ir.types import int8
import pytest
import numpy as np


def test_simple_loop():
    L, D = 8, 8
    
    def f1(A: int8[L, D]):
        for i, j in allo.grid(L, D): 
            A[i, j] += 1
            
    def f2(A: int8[L, D]):
        for i, j in allo.grid(L, D): 
            A[i, j] -= 1
            
    unified = unify_kernels(f1, f2)
    print(unified)
    llvm_mod = allo.LLVMModule(unified, "f1_f2_unified")
    allo_A = np.zeros((L, D), dtype=np.int8)
    np_A = allo_A.copy()
    np_A_add = allo_A + 2
    llvm_mod(allo_A, np.array([0, 0], dtype=np.int8))
    np.testing.assert_allclose(allo_A, np_A_add, atol=1e-3)
    llvm_mod(allo_A, np.array([1, 1], dtype=np.int8))
    np.testing.assert_allclose(np_A, allo_A, atol=1e-3)
    

if __name__ == "__main__":
    pytest.main([__file__])
    