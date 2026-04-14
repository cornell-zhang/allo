# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import allo
from allo.ir.types import int32, Stream
import allo.dataflow as df

def test_stream_try_get():
    # Test valid/ready stream primitives at IR level
    @df.region()
    def top():
        req_valid: Stream[int32, 2][1]
        
        @df.kernel(mapping=[1])
        def compute_tile():
            msg, has_req = req_valid[0].try_get()
            if has_req:
                pass
        
    mod = allo.customize(top)
    ir = str(mod.module)
    print(ir)
    assert "allo.stream_try_get" in ir, "Expected allo.stream_try_get in MLIR"

def test_stream_try_put():
    @df.region()
    def top():
        req_valid: Stream[int32, 2][1]
        
        @df.kernel(mapping=[1])
        def memory_tile():
            success = req_valid[0].try_put(1)
            if success:
                pass
        
    mod = allo.customize(top)
    ir = str(mod.module)
    assert "allo.stream_try_put" in ir, "Expected allo.stream_try_put in MLIR"

def test_stream_empty_full():
    @df.region()
    def top():
        req_valid: Stream[int32, 2][1]
        
        @df.kernel(mapping=[1])
        def memory_tile():
            while req_valid[0].full() or req_valid[0].empty():
                pass
        
    mod = allo.customize(top)
    ir = str(mod.module)
    assert "allo.stream_full" in ir, "Expected allo.stream_full in MLIR"
    assert "allo.stream_empty" in ir, "Expected allo.stream_empty in MLIR"

if __name__ == "__main__":
    test_stream_try_get()
    test_stream_try_put()
    test_stream_empty_full()
    print("Level 1 Stream IR Tests Passed!")
