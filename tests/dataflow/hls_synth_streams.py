# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
HLS Synthesis Script: Blocking vs Non-Blocking Stream Primitives
================================================================
Runs actual Vitis HLS synthesis to get LUT/FF/II numbers.
Usage: conda run -n allo python tests/dataflow/hls_synth_streams.py
"""
from __future__ import annotations
import os
import sys
import json
import numpy as np
import allo
import allo.dataflow as df
from allo.ir.types import int32, Stream, int1
from allo.backend import hls

# ─── Design Definitions ───────────────────────────────────────────────────────

@df.region()
def top_blocking(out: int32[4]):
    """Baseline: blocking put/get pair."""
    S: Stream[int32, 4][1]

    @df.kernel(mapping=[1])
    def producer():
        for i in range(4):
            S[0].put(i * 10)

    @df.kernel(mapping=[1], args=[out])
    def consumer(out_buf: int32[4]):
        for i in range(4):
            out_buf[i] = S[0].get()


@df.region()
def top_nonblocking(out: int32[4]):
    """Non-blocking try_put/try_get pair (spin-until-success)."""
    S: Stream[int32, 4][1]

    @df.kernel(mapping=[1])
    def producer_nb():
        for i in range(4):
            while not S[0].try_put(i * 10):
                pass

    @df.kernel(mapping=[1], args=[out])
    def consumer_nb(out_buf: int32[4]):
        for i in range(4):
            val: int32 = 0
            ok: int1 = 0
            while ok == 0:
                val, ok = S[0].try_get()
            out_buf[i] = val


@df.region()
def top_status_flags(out: int32[4]):
    """empty()/full() status flags test."""
    S: Stream[int32, 1][1]

    @df.kernel(mapping=[1], args=[out])
    def test_logic(out_buf: int32[4]):
        e0: int1 = S[0].empty()
        f0: int1 = S[0].full()
        out_buf[0] = 1 if e0 else 0   # expect 1
        out_buf[1] = 1 if f0 else 0   # expect 0
        ok: int1 = S[0].try_put(99)
        out_buf[2] = 1 if ok else 0   # expect 1
        e1: int1 = S[0].empty()
        f1: int1 = S[0].full()
        out_buf[3] = 1 if e1 else 0   # expect 0 (not empty after put)


# ─── CSIM: Functional Verification ────────────────────────────────────────────

def run_csim(region_fn, args_np, project_dir, desc):
    """Run C-simulation for a df.region and verify it compiles/runs."""
    if not hls.is_available("vitis_hls"):
        print(f"[SKIP] vitis_hls not available, skipping CSIM for {desc}")
        return None
    print(f"\n[CSIM] {desc} → {project_dir}")
    mod = df.build(region_fn, target="vitis_hls", mode="csim",
                   project=project_dir, wrap_io=True)
    mod(*args_np)
    return mod


# ─── CSYN: HLS Synthesis ──────────────────────────────────────────────────────

def run_csyn(region_fn, project_dir, desc):
    """Run Vitis HLS synthesis and return resource dictionary."""
    if not hls.is_available("vitis_hls"):
        print(f"[SKIP] vitis_hls not available, skipping CSYN for {desc}")
        return None
    print(f"\n[CSYN] {desc} → {project_dir}")
    mod = df.build(region_fn, target="vitis_hls", mode="csyn",
                   project=project_dir)
    mod()  # runs vitis_hls -f run.tcl
    # Parse synthesis report
    from allo.backend.report import parse_xml
    try:
        result = parse_xml(project_dir, "Vitis HLS",
                           top=mod.top_func_name, print_flag=True)
        return result
    except Exception as e:
        print(f"  [WARN] Could not parse report: {e}")
        return None


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    base = os.path.abspath("hls_projects")
    os.makedirs(base, exist_ok=True)

    results = {}

    # ── CSIM: verify correct outputs ──────────────────────────────────────────
    print("\n" + "="*60)
    print("  C-SIMULATION (Functional Verification)")
    print("="*60)

    np_out = np.zeros(4, dtype=np.int32)
    run_csim(top_blocking, [np_out], f"{base}/blocking_csim.prj",
             "Blocking put/get")
    np.testing.assert_array_equal(np_out, [0, 10, 20, 30])
    print(f"  blocking expected [0,10,20,30], got {np_out} ✓")

    np_out_nb = np.zeros(4, dtype=np.int32)
    run_csim(top_nonblocking, [np_out_nb], f"{base}/nonblocking_csim.prj",
             "Non-blocking try_put/try_get")
    np.testing.assert_array_equal(np_out_nb, [0, 10, 20, 30])
    print(f"  non-blocking expected [0,10,20,30], got {np_out_nb} ✓")

    np_flags = np.zeros(4, dtype=np.int32)
    run_csim(top_status_flags, [np_flags], f"{base}/status_csim.prj",
             "empty()/full() status flags")
    np.testing.assert_array_equal(np_flags, [1, 0, 1, 0])
    print(f"  flags expected [1,0,1,0], got {np_flags} ✓")

    # ── CSYN: Synthesis reports ────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  HLS SYNTHESIS (Resource / Timing Reports)")
    print("="*60)

    r_blocking = run_csyn(top_blocking, f"{base}/blocking_csyn.prj",
                          "Blocking put/get")
    results["blocking"] = r_blocking

    r_nonblocking = run_csyn(top_nonblocking, f"{base}/nonblocking_csyn.prj",
                             "Non-blocking try_put/try_get")
    results["nonblocking"] = r_nonblocking

    r_status = run_csyn(top_status_flags, f"{base}/status_csyn.prj",
                        "empty()/full() status flags")
    results["status_flags"] = r_status

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    for name, r in results.items():
        if r:
            print(f"\n  {name}:")
            for k, v in r.items():
                print(f"    {k}: {v}")

    # Save JSON for docstring updating
    summary_path = f"{base}/stream_synth_summary.json"
    with open(summary_path, "w") as f:
        # Convert to serializable format
        serializable = {}
        for name, r in results.items():
            if r:
                serializable[name] = {k: str(v) for k, v in r.items()}
        json.dump(serializable, f, indent=2)
    print(f"\n[INFO] Report saved to {summary_path}")


if __name__ == "__main__":
    main()
