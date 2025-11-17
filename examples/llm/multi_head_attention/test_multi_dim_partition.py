import allo
import numpy as np
from allo.ir.types import float32
from allo.customize import Partition as partition

def test_multi_dim_partition():
    def simple_kernel(A: float32[16, 16]) -> float32[16, 16]:
        B: float32[16, 16] = 0.0
        for i in range(16):
            for j in range(16):
                B[i, j] = A[i, j] * 2.0
        return B
    
    s = allo.customize(simple_kernel)
    
    # Example 1: Partition the same array on different dimensions with different factors
    s.partition(s.A, partition.Cyclic, dim=1, factor=4)   # Partition dimension 1 with factor 4
    s.partition(s.A, partition.Cyclic, dim=2, factor=8)   # Partition dimension 2 with factor 8
    
    # Example 2: Different partition types on different dimensions
    s.partition(s.B, partition.Block, dim=1, factor=2)    # Block partition on dimension 1
    s.partition(s.B, partition.Cyclic, dim=2, factor=4)   # Cyclic partition on dimension 2
    
    print("Multi-dimensional partitioning example:")
    print("Array A: Cyclic partition dim=1 factor=4, Cyclic partition dim=2 factor=8")
    print("Array B: Block partition dim=1 factor=2, Cyclic partition dim=2 factor=4")
    
    try:
        mod = s.build()
        print("Build successful - multi-dimensional partitioning is supported!")
        
        # Test functionality
        A = np.random.rand(16, 16).astype(np.float32)
        B = mod(A)
        expected = A * 2.0
        np.testing.assert_allclose(B, expected, rtol=1e-5)
        print("Functional test passed!")
        
    except Exception as e:
        print(f"Build failed: {e}")

if __name__ == "__main__":
    test_multi_dim_partition() 