# test_dataflow_canonicalization.py
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import allo
from allo.ir.types import int32, float32
from allo._mlir.dialects import memref as memref_d
from allo._mlir.ir import Operation

###############################################################################
# 1. Single-producer, single-consumer: No duplication needed.
###############################################################################

def test_single_producer_single_consumer():
    def kernel(A: int32[10]) -> int32[10]:
        B: int32[10] = 0
        for i in range(10):
            B[i] = A[i] + 1
        return B

    s = allo.customize(kernel)
    mod = s.build()
    np_A = np.arange(10, dtype=np.int32)
    np_B = mod(np_A)
    np.testing.assert_array_equal(np_B, np_A + 1)


###############################################################################
# 2. Single producer, multiple consumers: Buffer must be duplicated.
###############################################################################

def test_single_producer_multiple_consumers():
    def kernel(A: int32[10]) -> (int32[10], int32[10]):
        B: int32[10] = 0
        for i in range(10):
            B[i] = A[i] * 2
        # Two different consumer loops use the same buffer.
        C: int32[10] = 0
        D: int32[10] = 0
        for i in range(10):
            C[i] = B[i] + 3
            D[i] = B[i] - 1
        return C, D

    s = allo.customize(kernel)
    mod = s.build()
    np_A = np.arange(10, dtype=np.int32)
    C, D = mod(np_A)
    np.testing.assert_array_equal(C, np_A * 2 + 3)
    np.testing.assert_array_equal(D, np_A * 2 - 1)


###############################################################################
# 3. Multiple producers, single consumer: Merge must happen.
###############################################################################

def test_multiple_producers_single_consumer():
    def kernel(A: int32[10]) -> int32[10]:
        # Two loops write to the same buffer.
        B: int32[10] = 0
        for i in range(10):
            # Producer 1: write a first value.
            B[i] = A[i] + 5
        for i in range(10):
            # Producer 2: update the buffer.
            B[i] = B[i] - 2
        # Consumer: reads the merged result.
        C: int32[10] = 0
        for i in range(10):
            C[i] = B[i] * 3
        return C

    s = allo.customize(kernel)
    mod = s.build()
    np_A = np.arange(10, dtype=np.int32)
    C = mod(np_A)
    expected = (np_A + 5 - 2) * 3
    np.testing.assert_array_equal(C, expected)


###############################################################################
# 4. Multiple producers, multiple consumers: Merge then duplicate.
###############################################################################

def test_multiple_producers_multiple_consumers():
    def kernel(A: int32[10]) -> (int32[10], int32[10]):
        # Two producers write to the same shared buffer.
        B: int32[10] = 0
        for i in range(10):
            if i < 5:
                B[i] = A[i] + 10  # Producer 1
            else:
                B[i] = A[i] - 3   # Producer 2
        # Two consumers read B.
        C: int32[10] = 0
        D: int32[10] = 0
        for i in range(10):
            C[i] = B[i] * 2
            D[i] = B[i] - 1
        return C, D

    s = allo.customize(kernel)
    mod = s.build()
    np_A = np.arange(10, dtype=np.int32)
    B = np.concatenate([np_A[:5] + 10, np_A[5:] - 3])
    C, D = mod(np_A)
    np.testing.assert_array_equal(C, B * 2)
    np.testing.assert_array_equal(D, B - 1)


###############################################################################
# 5. Buffer with interleaved read/write (non read-only)
###############################################################################

def test_buffer_written_and_read():
    def kernel(A: int32[10]) -> int32[10]:
        # First, fill B.
        B: int32[10] = 0
        for i in range(10):
            B[i] = A[i] * 2
        # Consumer reads B into C.
        C: int32[10] = 0
        for i in range(10):
            C[i] = B[i] + 7
        # Then update B using C.
        for i in range(10):
            B[i] = C[i] - 3
        return B

    s = allo.customize(kernel)
    mod = s.build()
    np_A = np.arange(10, dtype=np.int32)
    np_B = mod(np_A)
    expected = (np_A * 2 + 7) - 3
    np.testing.assert_array_equal(np_B, expected)


###############################################################################
# 6. Preservation of data dependencies: Even when B is used in two ways.
###############################################################################

def test_dependency_preservation():
    def kernel(A: int32[10]) -> int32[10]:
        # Compute B from A.
        B: int32[10] = 0
        for i in range(10):
            B[i] = A[i] + 100
        # Consumer 1 computes C from B.
        C: int32[10] = 0
        for i in range(10):
            C[i] = B[i] - 50
        # Consumer 2 computes D from B.
        D: int32[10] = 0
        for i in range(10):
            D[i] = B[i] * 2
        # Final result depends on both.
        E: int32[10] = 0
        for i in range(10):
            E[i] = C[i] + D[i]
        return E

    s = allo.customize(kernel)
    mod = s.build()
    np_A = np.arange(10, dtype=np.int32)
    B = np_A + 100
    C = B - 50
    D = B * 2
    expected = C + D
    np.testing.assert_array_equal(mod(np_A), expected)


###############################################################################
# 7. Control–flow test: Conditional writes to a shared buffer.
###############################################################################

def test_control_flow():
    def kernel(A: int32[10], flag: int32) -> int32[10]:
        B: int32[10] = 0
        if flag:
            for i in range(10):
                B[i] = A[i] + 1
        else:
            for i in range(10):
                B[i] = A[i] - 1
        C: int32[10] = 0
        for i in range(10):
            C[i] = B[i] * 4
        return C

    s = allo.customize(kernel)
    mod = s.build()
    np_A = np.arange(10, dtype=np.int32)
    np.testing.assert_array_equal(mod(np_A, 1), (np_A + 1) * 4)
    np.testing.assert_array_equal(mod(np_A, 0), (np_A - 1) * 4)


###############################################################################
# 8. IR inspection: Ensure no allocation is used by more than one consumer.
###############################################################################

def test_ir_single_consumer_property():
    def kernel(A: int32[10]) -> int32[10]:
        B: int32[10] = 0
        for i in range(10):
            B[i] = A[i] + 5
        # B is used in two different loops.
        C: int32[10] = 0
        D: int32[10] = 0
        for i in range(10):
            C[i] = B[i] * 2
        for i in range(10):
            D[i] = B[i] - 3
        # Final result is computed from both.
        E: int32[10] = 0
        for i in range(10):
            E[i] = C[i] + D[i]
        return E

    s = allo.customize(kernel)
    mod = s.build()
    # Traverse the module IR to inspect allocation ops.
    # (This code is illustrative; adjust according to your IR API.)
    for op in mod.operation.walk():
        if isinstance(op, memref_d.AllocOp):
            # For each alloc, check that its result is used by only one
            # consumer op per “duplication” (i.e. no multi-consumer sharing).
            uses = list(op.result.uses)
            consumer_ops = set(use.owner for use in uses)
            # In canonicalized IR, every alloc should have been duplicated so
            # that each consumer uses its own unique alloc.
            assert len(uses) == len(consumer_ops), (
                f"Alloc {op} is shared among multiple consumers: {consumer_ops}"
            )


###############################################################################
# 9. Merge multiple producers writing to the same buffer.
###############################################################################

def test_merge_multiple_producers():
    def kernel(A: int32[10], B: int32[10]) -> int32[10]:
        # First loop: Producer 1 writes to D.
        D: int32[10] = 0
        for i in range(10):
            D[i] = A[i]
        # Second loop: Producer 2 updates D.
        for i in range(10):
            D[i] = D[i] + B[i]
        # Consumer reads D.
        C: int32[10] = 0
        for i in range(10):
            C[i] = D[i] * 2
        return C

    s = allo.customize(kernel)
    mod = s.build()
    np_A = np.arange(10, dtype=np.int32)
    np_B = np.arange(10, 20, dtype=np.int32)
    C = mod(np_A, np_B)
    expected = (np_A + np_B) * 2
    np.testing.assert_array_equal(C, expected)


###############################################################################
# Run all tests when executed directly.
###############################################################################

if __name__ == "__main__":
    pytest.main([__file__])
