from __future__ import annotations
import pytest
import allo
from allo.ir.types import int32, Stream
import allo.dataflow as df
import numpy as np

def test_stream_nb_sim():
    @df.region()
    def top_simple():
        S: Stream[int32, 2][1]
        Out: Stream[int32, 4][1]
        
        @df.kernel(mapping=[1])
        def producer():
            for i in range(4):
                while not S[0].try_put(i):
                    pass
        
        @df.kernel(mapping=[1])
        def consumer():
            for i in range(4):
                data, success = S[0].try_get()
                while not success:
                    data, success = S[0].try_get()
                Out[0].try_put(data)

    try:
        simulator = df.build(top_simple, target="simulator")
        print("Successfully built simulator for df.region")
        
        # Run simulation
        simulator()
        
        print("Simulation finished successfully!")
    except Exception as e:
        print(f"Simulation failed: {e}")
        raise e

if __name__ == "__main__":
    test_stream_nb_sim()

