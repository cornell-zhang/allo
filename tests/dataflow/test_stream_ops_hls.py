import pytest
import allo
from allo.ir.types import int32, Stream
import allo.dataflow as df

def test_vhls_stream_nb():
    @df.region()
    def top():
        S0: Stream[int32, 2][1]
        S1: Stream[int32, 2][1]

        @df.kernel(mapping=[1])
        def kernel():
            # try_get
            data, success = S0[0].try_get()
            if success:
                # try_put
                ok = S1[0].try_put(data)
                if ok:
                    pass
            # empty/full
            if not S0[0].empty() and not S1[0].full():
                pass

    mod = allo.customize(top)
    hls_mod = mod.build(target="vhls")
    hls_code = hls_mod.hls_code
    
    assert ".read_nb(" in hls_code
    assert ".write_nb(" in hls_code
    assert ".empty()" in hls_code
    assert ".full()" in hls_code
    print("Level 2 VHLS Stream NB Test Passed!")

def test_tapa_stream_nb():
    @df.region()
    def top():
        S0: Stream[int32, 2][1]
        
        @df.kernel(mapping=[1])
        def kernel():
            data, success = S0[0].try_get()
            if success:
                S0[0].try_put(data)

    mod = allo.customize(top)
    hls_mod = mod.build(target="tapa")
    hls_code = hls_mod.hls_code
    
    assert ".try_read(" in hls_code
    assert ".try_write(" in hls_code
    print("Level 2 Tapa Stream NB Test Passed!")

if __name__ == "__main__":
    test_vhls_stream_nb()
    test_tapa_stream_nb()


