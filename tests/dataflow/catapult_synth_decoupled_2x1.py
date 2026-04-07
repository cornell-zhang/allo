"""
Catapult HLS synthesis script for decoupled mesh designs.

Designs
-------
    decoupled_2x1  — 1 MT → 2 CTs (non-blocking optional, blocking would work)
    arb_2to1       — 2 MTs → 1 CT (non-blocking STRICTLY required for arbitration)

Architecture: decoupled_2x1
----------------------------
    Memory Tile (MT)  ←valid-ready handshake→  Compute Tile 0 (CT0)
                      ←valid-ready handshake→  Compute Tile 1 (CT1)

Architecture: arb_2to1 (many-to-one arbitration)
-------------------------------------------------
    Memory Tile A (MT0) ──req──►  ┌──────────────────┐
                        ◄─grant─  │                  │
                        ──data──► │  Compute Tile    │
                        ◄─data──  │  (arbitrator)    │
                                  │                  │
    Memory Tile B (MT1) ──req──►  │  polls BOTH req  │
                        ◄─grant─  │  channels with   │
                        ──data──► │  try_get / empty  │
                        ◄─data──  └──────────────────┘

    WHY non-blocking is STRICTLY necessary here:
    The CT has two request channels (one from each MT). MT0 and MT1 are
    independent concurrent processes — neither knows the other's timing.
    If CT does `req_from_mt[0].get()` (blocking), it commits to waiting
    on MT0. If MT1 sends first while MT0 hasn't, CT deadlocks.
    Non-blocking `empty()` + `try_get()` lets CT poll both channels and
    serve whichever MT sends first.

Non-blocking stream ops used:
    try_put  → ac_channel::nb_write()
    try_get  → ac_channel::nb_read()
    empty()  → ac_channel::empty()

Synthesis modes
---------------
    "codegen"  — emit kernel.cpp only (no Catapult required)
    "csyn"     — run Catapult synthesis, generate Verilog
    "ppa"      — run synthesis + parse area/latency/power reports
                 (hierarchical: per-MT, per-CT, interconnect overhead)

Usage
-----
    # Quick codegen check (no Catapult needed):
    conda run -n allo python tests/dataflow/catapult_synth_decoupled_2x1.py

    # Arbitration design codegen:
    conda run -n allo python tests/dataflow/catapult_synth_decoupled_2x1.py --design arb_2to1

    # Full synthesis (requires Catapult + MGC_HOME):
    conda run -n allo python tests/dataflow/catapult_synth_decoupled_2x1.py --mode csyn

    # PPA extraction:
    conda run -n allo python tests/dataflow/catapult_synth_decoupled_2x1.py --mode ppa
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import allo
import allo.dataflow as df
from allo.ir.types import int32, float32, int1, Stream, stateful

# ─── Design constants ─────────────────────────────────────────────────────────
BURST_SIZE = 16
N_CTS = 2
N_MTS = 2

MSG_WRITE   = 1
MSG_READ    = 2
MSG_COMPUTE = 4
MSG_GRANT   = 3
MSG_REQ     = 5   # Generic request (used by arb_2to1)

# ─── 2×1 decoupled mesh (same as test_decoupled_mesh.py::top_decoupled_2x1) ──

@df.region()
def top_decoupled_2x1(
    in0:  float32[BURST_SIZE],
    in1:  float32[BURST_SIZE],
    out0: float32[BURST_SIZE],
    out1: float32[BURST_SIZE],
):
    # Per-CT handshaked control streams (MT→CT: request; CT→MT: grant)
    req_valid:   Stream[int32, 2][N_CTS]
    grant_ready: Stream[int32, 2][N_CTS]
    # Per-CT unconditional data burst streams
    data_mt2ct: Stream[float32, BURST_SIZE][N_CTS]
    data_ct2mt: Stream[float32, BURST_SIZE][N_CTS]

    @df.kernel(mapping=[1], args=[in0, in1, out0, out1])
    def memory_tile_2x1(
        in0_p:  float32[BURST_SIZE],
        in1_p:  float32[BURST_SIZE],
        out0_p: float32[BURST_SIZE],
        out1_p: float32[BURST_SIZE],
    ):
        # ── Phase 1: WRITE to CT0 ──────────────────────────────────────────
        sent: int1 = 0
        while sent == 0:
            sent = req_valid[0].try_put(MSG_WRITE)
        ack: int1 = 0
        while ack == 0:
            grant_val, ack = grant_ready[0].try_get()
        for i in range(BURST_SIZE):
            data_mt2ct[0].put(in0_p[i])

        # ── Phase 1: WRITE to CT1 ──────────────────────────────────────────
        sent = 0
        while sent == 0:
            sent = req_valid[1].try_put(MSG_WRITE)
        ack = 0
        while ack == 0:
            grant_val, ack = grant_ready[1].try_get()
        for i in range(BURST_SIZE):
            data_mt2ct[1].put(in1_p[i])

        # ── Phase 2: COMPUTE trigger to CT0 ───────────────────────────────
        sent = 0
        while sent == 0:
            sent = req_valid[0].try_put(MSG_COMPUTE)
        ack = 0
        while ack == 0:
            grant_val, ack = grant_ready[0].try_get()

        # ── Phase 2: COMPUTE trigger to CT1 ───────────────────────────────
        sent = 0
        while sent == 0:
            sent = req_valid[1].try_put(MSG_COMPUTE)
        ack = 0
        while ack == 0:
            grant_val, ack = grant_ready[1].try_get()

        # ── Phase 3: READ from CT0 ────────────────────────────────────────
        sent = 0
        while sent == 0:
            sent = req_valid[0].try_put(MSG_READ)
        ack = 0
        while ack == 0:
            grant_val, ack = grant_ready[0].try_get()
        for i in range(BURST_SIZE):
            out0_p[i] = data_ct2mt[0].get()

        # ── Phase 3: READ from CT1 ────────────────────────────────────────
        sent = 0
        while sent == 0:
            sent = req_valid[1].try_put(MSG_READ)
        ack = 0
        while ack == 0:
            grant_val, ack = grant_ready[1].try_get()
        for i in range(BURST_SIZE):
            out1_p[i] = data_ct2mt[1].get()

    @df.kernel(mapping=[N_CTS])
    def compute_tile_2x1():
        """Each instance handles one req_valid/grant_ready/data channel."""
        id = df.get_pid()
        spad: float32[BURST_SIZE] @ stateful = 0.0

        req_count: int32 = 0
        while req_count < 3:
            has_req: int1 = 0
            msg_type: int32 = 0

            if not req_valid[id].empty():
                msg_type, has_req = req_valid[id].try_get()

            if has_req == 1:
                if msg_type == MSG_WRITE:
                    grant_sent: int1 = 0
                    while grant_sent == 0:
                        grant_sent = grant_ready[id].try_put(MSG_GRANT)
                    for i in range(BURST_SIZE):
                        spad[i] = data_mt2ct[id].get()

                elif msg_type == MSG_COMPUTE:
                    for i in range(BURST_SIZE):
                        spad[i] = spad[i] + 1.0
                    grant_sent = 0
                    while grant_sent == 0:
                        grant_sent = grant_ready[id].try_put(MSG_GRANT)

                elif msg_type == MSG_READ:
                    grant_sent = 0
                    while grant_sent == 0:
                        grant_sent = grant_ready[id].try_put(MSG_GRANT)
                    for i in range(BURST_SIZE):
                        data_ct2mt[id].put(spad[i])

                req_count += 1


# ─── 2-to-1 arbitration: 2 MTs → 1 shared CT (non-blocking REQUIRED) ────────
#
# This design CANNOT be implemented with blocking FIFOs.
#
# The CT has two incoming request channels. MT0 and MT1 are independent
# concurrent processes — the CT cannot know which will send first.
# A blocking get() on either channel would deadlock if the other MT
# sends first:
#
#   DEADLOCK SCENARIO (with blocking):
#     CT: req_from_mt[0].get()   ← blocks waiting for MT0
#     MT1: req_from_mt[1].put()  ← MT1 sends, but CT is stuck on [0]
#     MT0: (still computing)     ← MT0 hasn't sent yet
#     → CT hangs forever, MT1's message sits unread
#
# With non-blocking, CT polls both channels each iteration:
#     while served < 2:
#         if not req_from_mt[0].empty(): handle MT0
#         if not req_from_mt[1].empty(): handle MT1
#
# Protocol per MT:
#   1. MT sends request on req_from_mt[id] (try_put spin)
#   2. MT waits for grant on grant_to_mt[id] (try_get spin)
#   3. MT burst-sends data on data_to_ct[id]
#   4. MT burst-receives result on data_from_ct[id]

@df.region()
def top_arb_2to1(
    in0:  float32[BURST_SIZE],
    in1:  float32[BURST_SIZE],
    out0: float32[BURST_SIZE],
    out1: float32[BURST_SIZE],
):
    # Host ↔ MT streams (host_io distributes data to/from the two MTs)
    host_to_mt: Stream[float32, BURST_SIZE][N_MTS]
    mt_to_host: Stream[float32, BURST_SIZE][N_MTS]
    # Per-MT control streams (MT→CT request, CT→MT grant)
    req_from_mt:  Stream[int32, 2][N_MTS]
    grant_to_mt:  Stream[int32, 2][N_MTS]
    # Per-MT data streams
    data_to_ct:   Stream[float32, BURST_SIZE][N_MTS]
    data_from_ct: Stream[float32, BURST_SIZE][N_MTS]

    # ── Host I/O: distributes inputs to MTs, collects results ────────────
    @df.kernel(mapping=[1], args=[in0, in1, out0, out1])
    def host_io_arb(
        in0_p:  float32[BURST_SIZE],
        in1_p:  float32[BURST_SIZE],
        out0_p: float32[BURST_SIZE],
        out1_p: float32[BURST_SIZE],
    ):
        for i in range(BURST_SIZE):
            host_to_mt[0].put(in0_p[i])
        for i in range(BURST_SIZE):
            host_to_mt[1].put(in1_p[i])
        for i in range(BURST_SIZE):
            out0_p[i] = mt_to_host[0].get()
        for i in range(BURST_SIZE):
            out1_p[i] = mt_to_host[1].get()

    # ── MTs: 2 instances, each gets data from host_io, sends to shared CT
    @df.kernel(mapping=[N_MTS])
    def memory_tile_arb():
        id = df.get_pid()
        in_buf: float32[BURST_SIZE] @ stateful = 0.0
        for i in range(BURST_SIZE):
            in_buf[i] = host_to_mt[id].get()
        sent: int1 = 0
        while sent == 0:
            sent = req_from_mt[id].try_put(MSG_REQ)
        ack: int1 = 0
        while ack == 0:
            grant_val, ack = grant_to_mt[id].try_get()
        for i in range(BURST_SIZE):
            data_to_ct[id].put(in_buf[i])
        for i in range(BURST_SIZE):
            in_buf[i] = data_from_ct[id].get()
        for i in range(BURST_SIZE):
            mt_to_host[id].put(in_buf[i])

    # ── Shared CT: arbitrates between MT0 and MT1 ────────────────────────
    #
    # This kernel is the reason non-blocking is strictly necessary.
    # It must poll BOTH req_from_mt[0] and req_from_mt[1] because
    # it cannot predict which MT will send first.
    @df.kernel(mapping=[1])
    def compute_tile_arb():
        spad: float32[BURST_SIZE] @ stateful = 0.0

        served: int32 = 0
        while served < N_MTS:
            # ── Poll MT0 ────────────────────────────────────────────
            has_req_0: int1 = 0
            msg_0: int32 = 0
            if not req_from_mt[0].empty():
                msg_0, has_req_0 = req_from_mt[0].try_get()

            if has_req_0 == 1:
                gs_0: int1 = 0
                while gs_0 == 0:
                    gs_0 = grant_to_mt[0].try_put(MSG_GRANT)
                for i in range(BURST_SIZE):
                    spad[i] = data_to_ct[0].get()
                for i in range(BURST_SIZE):
                    spad[i] = spad[i] + 1.0
                for i in range(BURST_SIZE):
                    data_from_ct[0].put(spad[i])
                served = served + 1

            # ── Poll MT1 ────────────────────────────────────────────
            has_req_1: int1 = 0
            msg_1: int32 = 0
            if not req_from_mt[1].empty():
                msg_1, has_req_1 = req_from_mt[1].try_get()

            if has_req_1 == 1:
                gs_1: int1 = 0
                while gs_1 == 0:
                    gs_1 = grant_to_mt[1].try_put(MSG_GRANT)
                for i in range(BURST_SIZE):
                    spad[i] = data_to_ct[1].get()
                for i in range(BURST_SIZE):
                    spad[i] = spad[i] + 1.0
                for i in range(BURST_SIZE):
                    data_from_ct[1].put(spad[i])
                served = served + 1


# ─── Synthesis / codegen helpers ─────────────────────────────────────────────

# ─── Design registry ─────────────────────────────────────────────────────────

DESIGNS = {
    "decoupled_2x1": {
        "region": top_decoupled_2x1,
        "project": "catapult_decoupled_2x1.prj",
        "sub_funcs": [
            "memory_tile_2x1_0",
            "compute_tile_2x1_0",
            "compute_tile_2x1_1",
        ],
    },
    "arb_2to1": {
        "region": top_arb_2to1,
        "project": "catapult_arb_2to1.prj",
        "sub_funcs": [
            "host_io_arb_0",
            "memory_tile_arb_0",
            "memory_tile_arb_1",
            "compute_tile_arb_0",
        ],
    },
}


def run_codegen(design: dict, project: str) -> str:
    """Emit Catapult HLS C++ kernel; return path to kernel.cpp."""
    region = design["region"]
    print(f"[codegen] Building {region.__name__} with target=catapult ...")
    mod = df.build(
        region,
        target="catapult",
        project=project,
        mode="csyn",
        configs={
            "preserve_hierarchy": True,
            "sub_funcs": design["sub_funcs"],
        },
    )
    kernel_path = os.path.join(project, "kernel.cpp")
    print(f"[codegen] kernel.cpp written to: {kernel_path}")
    if os.path.exists(kernel_path):
        with open(kernel_path) as f:
            snippet = f.read()[:2000]
        print("─── kernel.cpp (first 2000 chars) ───")
        print(snippet)
        print("─────────────────────────────────────")
    return kernel_path


def run_synthesis(design: dict, project: str, mode: str) -> dict:
    """Run Catapult synthesis and return PPA stats dict."""
    region = design["region"]
    print(f"[synthesis] mode={mode}, project={project}, design={region.__name__}")
    t0 = time.time()
    mod = df.build(
        region,
        target="catapult",
        project=project,
        mode=mode,
        configs={
            "preserve_hierarchy": True,
            "sub_funcs": design["sub_funcs"],
        },
    )
    stats = mod()
    elapsed = time.time() - t0
    print(f"[synthesis] completed in {elapsed:.1f}s")
    return stats or {}


def verify_simulator(design: dict):
    """Quick functional verification on the Allo simulator before HLS."""
    region = design["region"]
    print(f"[verify] Running simulator functional test for {region.__name__} ...")
    sim = df.build(region, target="simulator")
    np_in0  = np.arange(BURST_SIZE, dtype=np.float32)
    np_in1  = np.arange(BURST_SIZE, dtype=np.float32) * 2.0
    np_out0 = np.zeros(BURST_SIZE, dtype=np.float32)
    np_out1 = np.zeros(BURST_SIZE, dtype=np.float32)
    sim(np_in0, np_in1, np_out0, np_out1)
    np.testing.assert_allclose(np_out0, np_in0 + 1.0, rtol=1e-5)
    np.testing.assert_allclose(np_out1, np_in1 + 1.0, rtol=1e-5)
    print("[verify] PASSED — both outputs are input + 1.0")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Catapult HLS synthesis for decoupled mesh designs"
    )
    parser.add_argument(
        "--design",
        choices=list(DESIGNS.keys()),
        default="decoupled_2x1",
        help=(
            "decoupled_2x1: 1 MT → 2 CTs (default); "
            "arb_2to1: 2 MTs → 1 CT (arbitration, non-blocking required)"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["codegen", "csyn", "ppa"],
        default="codegen",
        help=(
            "codegen: emit kernel.cpp only (default); "
            "csyn: run Catapult synthesis; "
            "ppa: synthesis + hierarchical PPA report"
        ),
    )
    parser.add_argument(
        "--project",
        default=None,
        help="Catapult project directory name (default: per-design)",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip simulator verification step",
    )
    args = parser.parse_args()

    design = DESIGNS[args.design]
    project = args.project or design["project"]

    if not args.no_verify:
        verify_simulator(design)

    if args.mode == "codegen":
        run_codegen(design, project)
        print("\n[done] Codegen complete. Inspect kernel.cpp for ac_channel API usage.")
        print("       Non-blocking ops should use .nb_write() / .nb_read() / .empty()")
    else:
        siemens_root = "/opt/siemens/catapult"
        mgc_home = os.environ.get("MGC_HOME", "")
        if not mgc_home and os.path.isdir(siemens_root):
            versions = sorted(
                [d for d in os.listdir(siemens_root)
                 if os.path.isdir(os.path.join(siemens_root, d))],
                reverse=True,
            )
            if versions:
                os.environ["MGC_HOME"] = os.path.join(siemens_root, versions[0])
                mgc_home = os.environ["MGC_HOME"]
                print(f"[info] Auto-detected Catapult: {mgc_home}")
        if not mgc_home:
            print(
                "ERROR: Catapult not found. Set MGC_HOME or install to /opt/siemens/catapult/.",
                file=sys.stderr,
            )
            sys.exit(1)
        stats = run_synthesis(design, project, args.mode)
        if args.mode == "ppa" and stats:
            hier = stats.get("hierarchical", {})
            if hier:
                print()
                print("=== Hierarchical PPA Summary ===")
                print(hier.get("summary", "(no data)"))


if __name__ == "__main__":
    main()
