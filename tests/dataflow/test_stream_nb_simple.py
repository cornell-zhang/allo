# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Non-Blocking Stream Operation Tests
====================================
Tests for Allo's non-blocking stream primitives: try_put, try_get, empty, full.
These enable decoupled, handshaked communication between kernels without
the strict fixed-II (balanced send/receive) constraint of blocking FIFOs.

Architecture Context
--------------------
Standard valid-ready handshake over on-chip networks (AXI-S, NoC, etc.) maps
naturally to try_put (assert valid) + try_get (sample data when ready) patterns.
The complementary empty/full predicates expose backpressure signals.

HLS Cost Notes (Vitis HLS synthesis on U280, estimated, not measured here)
---------------------------------------------------------------------------
- empty(): Compiles to a combinational comparison of head/tail pointers.
  Cost: ~2 FF reads + 1-2 LUT comparator. No II penalty.
- full(): Same cost profile as empty().
- try_put(): Adds a conditional branch (is_not_full) around the write path.
  Compared to blocking put: same data write logic, plus one extra comparator
  and a MUX for the success bit. Depth-2 stream: ~4 FFs (head/tail + 2-entry buf).
- try_get(): Symmetric to try_put(). One extra comparator + MUX for success bit.

Scalability Notes
-----------------
- Non-blocking ops add ~O(1) LUTs per stream (one comparator + branch).
- The success bit propagation through the control path is the main scheduling
  concern in HLS - use it only in control logic, not on the datapath critical path.
- For scalable mesh routing, the control (try_put/try_get) path stays thin;
  the data burst streams remain blocking for high-throughput, area-efficient transfer.
"""
from __future__ import annotations
import pytest
import allo
from allo.ir.types import int32, Stream, int1
import allo.dataflow as df
import numpy as np


# ---------------------------------------------------------------------------
# Test 1: Simulator - Producer / Consumer with try_put / try_get
# ---------------------------------------------------------------------------

def test_try_put_try_get_sim():
    """
    Two-kernel test: producer sends via try_put (spin-until-success),
    consumer receives via try_get (spin-until-success).
    Result is accumulated in an output stream then checked.
    """
    @df.region()
    def top_nb(out: int32[4]):
        S: Stream[int32, 4][1]
        res: Stream[int32, 4][1]

        @df.kernel(mapping=[1])
        def producer():
            for i in range(4):
                while not S[0].try_put(i * 10):
                    pass

        @df.kernel(mapping=[1], args=[out])
        def consumer(out_buf: int32[4]):
            for i in range(4):
                val: int32 = 0
                ok: int1 = 0
                while ok == 0:
                    val, ok = S[0].try_get()
                out_buf[i] = val

    sim = df.build(top_nb, target="simulator")
    np_out = np.zeros(4, dtype=np.int32)
    sim(np_out)
    np.testing.assert_array_equal(np_out, [0, 10, 20, 30])
    print("test_try_put_try_get_sim PASSED")


# ---------------------------------------------------------------------------
# Test 2: Simulator - empty() and full() status flags
# ---------------------------------------------------------------------------

def test_empty_full_sim():
    """
    Single-kernel test: verifies that empty()/full() return correct status.
    - Initially: empty=True, full=False
    - After try_put (depth-1 FIFO): empty=False, full=True
    """
    @df.region()
    def nb_status(out: int32[5]):
        S: Stream[int32, 1][1]

        @df.kernel(mapping=[1], args=[out])
        def test_logic(out_buf: int32[5]):
            # Initially empty, not full
            e0: int1 = S[0].empty()
            f0: int1 = S[0].full()
            out_buf[0] = 1 if e0 else 0  # expect 1
            out_buf[1] = 1 if f0 else 0  # expect 0

            # try_put one element into depth-1 stream
            success: int1 = S[0].try_put(42)
            out_buf[4] = 1 if success else 0  # expect 1

            # Now: not empty, full
            e1: int1 = S[0].empty()
            f1: int1 = S[0].full()
            out_buf[2] = 1 if e1 else 0  # expect 0
            out_buf[3] = 1 if f1 else 0  # expect 1

    sim = df.build(nb_status, target="simulator")
    np_out = np.zeros(5, dtype=np.int32)
    sim(np_out)
    assert np_out[0] == 1, f"empty() should be True initially, got {np_out[0]}"
    assert np_out[1] == 0, f"full() should be False initially, got {np_out[1]}"
    assert np_out[2] == 0, f"empty() should be False after put, got {np_out[2]}"
    assert np_out[3] == 1, f"full() should be True after put, got {np_out[3]}"
    assert np_out[4] == 1, f"try_put() should succeed, got {np_out[4]}"
    print("test_empty_full_sim PASSED")


# ---------------------------------------------------------------------------
# Test 3: HLS Code Generation - verify correct API lowering
# ---------------------------------------------------------------------------

def test_nb_ops_hls_codegen():
    """
    Verifies that non-blocking stream ops lower to correct Vitis HLS API calls:
      - try_get  -> stream.read_nb(val)   (non-blocking read)
      - try_put  -> stream.write_nb(val)  (non-blocking write)
      - empty()  -> stream.empty()
      - full()   -> stream.full()

    HLS Performance Notes (Vivado HLS / Vitis HLS):
      - .read_nb() and .write_nb() do NOT introduce pipeline stalls.
        They return a boolean success flag and complete in II=1 always.
      - Blocking .read() / .write() introduce FIFO handshake stalls (II varies).
      - Use non-blocking variants on control paths; blocking variants on data bursts.
      - Synthesis: each non-blocking op adds ~1 FIFO depth entry + 2 control FFs.
    """
    @df.region()
    def top_nb_hls():
        ctrl_in: Stream[int32, 2][1]
        ctrl_out: Stream[int32, 2][1]

        @df.kernel(mapping=[1])
        def relay():
            # Non-blocking receive
            val: int32 = 0
            ok: int1 = 0
            val, ok = ctrl_in[0].try_get()
            if ok:
                # Non-blocking send
                sent: int1 = ctrl_out[0].try_put(val + 1)
            # Status checks
            e: int1 = ctrl_in[0].empty()
            f: int1 = ctrl_out[0].full()
            if e or f:
                pass

    mod = allo.customize(top_nb_hls)
    hls_mod = mod.build(target="vhls")
    hls_code = hls_mod.hls_code

    assert ".read_nb(" in hls_code, "Expected .read_nb() for try_get"
    assert ".write_nb(" in hls_code, "Expected .write_nb() for try_put"
    assert ".empty()" in hls_code, "Expected .empty() status check"
    assert ".full()" in hls_code, "Expected .full() status check"
    print("test_nb_ops_hls_codegen PASSED")
    print("  HLS code snippet:")
    for line in hls_code.split('\n'):
        if any(kw in line for kw in ['.read_nb', '.write_nb', '.empty()', '.full()']):
            print("   ", line.strip())


# ---------------------------------------------------------------------------
# NOTE: CSIM / COSIM require Vitis HLS to be installed and configured.
# Run these manually with:
#   df.build(top, target="vitis_hls", mode="csim", project="nb_csim.prj")
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    test_empty_full_sim()
    test_try_put_try_get_sim()
    test_nb_ops_hls_codegen()
    print("\nAll test_stream_nb_simple tests PASSED!")
