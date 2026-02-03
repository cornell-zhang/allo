# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile
import pytest
import allo
from allo.ir.types import int32, float32

def test_catapult_partition():
    def partition_test(A: int32[10, 10]) -> int32[10, 10]:
        B: int32[10, 10]
        for i, j in allo.grid(10, 10):
            B[i, j] = A[i, j] + 1
        return B

    s = allo.customize(partition_test)
    s.partition(s.A, dim=1)
    s.partition(s.B, dim=1)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="catapult", mode="csyn", project=tmpdir)
        print(mod.hls_code)
        # Check that we do NOT emit Vivado HLS pragmas
        assert "#pragma HLS array_partition" not in mod.hls_code
        # For now, we assume implicit partitioning or handled via other means. 
        # Ideally we should check if generated code handles parallel access if unrolled.
        # But here we just check we don't emit wrong pragmas.

def test_catapult_parallel():
    def parallel_test(A: int32[10]) -> int32[10]:
        B: int32[10]
        for i in allo.grid(10):
            B[i] = A[i] * 2
        return B
        
    s = allo.customize(parallel_test)
    s.parallel("i")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="catapult", mode="csyn", project=tmpdir)
        print(mod.hls_code)
        # Check for unroll pragma which is often used for parallel loops in HLS?
        # Or does Catapult have a specific parallel directive?
        # Usually full unroll = parallel.
        assert "#pragma hls_unroll" in mod.hls_code

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
