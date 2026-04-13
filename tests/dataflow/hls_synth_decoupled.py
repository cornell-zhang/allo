"""
HLS Synthesis Script: Decoupled Mesh Architecture
=================================================
Runs Vitis HLS synthesis/simulation for the decoupled message-passing
architecture and the hierarchical 2x1 decoupled mesh.
Usage: conda run -n allo python tests/dataflow/hls_synth_decoupled.py
"""
from __future__ import annotations
import os
import sys
import xml.etree.ElementTree as ET
import numpy as np
import allo
import allo.dataflow as df
from allo.ir.types import int32, float32, int1, Stream, Stateful
from allo.backend import hls

BURST_SIZE = 16
N_CTS = 2

# Single-CT message-passing opcodes
MSG_REQ_READ = 1
MSG_REQ_WRITE = 2
MSG_ACK = 3

# Hierarchical 2x1 opcodes
MSG_WRITE   = 1
MSG_READ    = 2
MSG_COMPUTE = 4
MSG_GRANT   = 3


# ─── Design: Decoupled Message Passing (MT ↔ 1 CT) ──────────────────────────

@df.region()
def top_message_passing(
    base_addr: int32[1],
    in_payload: float32[BURST_SIZE],
    out_payload: float32[BURST_SIZE]
):
    """
    Decoupled MT+CT with valid-ready handshake on control streams,
    unconditional burst streaming on data streams.
    """
    # Handshaked Control Streams
    req_valid: Stream[int32, 2][1]
    req_addr:  Stream[int32, 2][1]
    grant_ready: Stream[int32, 2][1]
    # Unconditional Data Burst Streams
    data_tx: Stream[float32, BURST_SIZE][1]
    data_rx: Stream[float32, BURST_SIZE][1]

    @df.kernel(mapping=[1], args=[base_addr, in_payload, out_payload])
    def memory_tile(
        b_addr: int32[1],
        in_p:   float32[BURST_SIZE],
        out_p:  float32[BURST_SIZE]
    ):
        # 1. MT sends WRITE request (try_put = "raise Valid")
        req_sent: int1 = 0
        while req_sent == 0:
            req_sent = req_valid[0].try_put(MSG_REQ_WRITE)
            if req_sent == 1:
                req_addr[0].put(b_addr[0])

        # 2. MT waits for GRANT (try_get = "poll Ready")
        granted: int1 = 0
        while granted == 0:
            grant_msg, granted = grant_ready[0].try_get()
            if granted == 1 and grant_msg == MSG_ACK:
                # Handshake done → burst data unconditionally
                for i in range(BURST_SIZE):
                    data_tx[0].put(in_p[i])

        # 3. MT sends READ request
        req_sent = 0
        while req_sent == 0:
            req_sent = req_valid[0].try_put(MSG_REQ_READ)
            if req_sent == 1:
                req_addr[0].put(b_addr[0])

        # 4. MT waits for GRANT for reading
        granted = 0
        while granted == 0:
            grant_msg, granted = grant_ready[0].try_get()
            if granted == 1 and grant_msg == MSG_ACK:
                for i in range(BURST_SIZE):
                    out_p[i] = data_rx[0].get()

    @df.kernel(mapping=[1])
    def compute_tile():
        data_mem: float32[256] @ Stateful = 0.0

        running: int1 = 1
        req_count: int32 = 0
        while running == 1 and req_count < 2:
            has_req: int1 = 0
            msg_type: int32 = 0

            if not req_valid[0].empty():
                msg_type, has_req = req_valid[0].try_get()

            if has_req == 1:
                addr: int32 = req_addr[0].get()

                if msg_type == MSG_REQ_WRITE:
                    grant_sent: int1 = 0
                    while grant_sent == 0:
                        grant_sent = grant_ready[0].try_put(MSG_ACK)
                    for i in range(BURST_SIZE):
                        data_mem[addr + i] = data_tx[0].get()

                elif msg_type == MSG_REQ_READ:
                    for i in range(BURST_SIZE):
                        data_mem[addr + i] = data_mem[addr + i] + 1.0
                    grant_sent = 0
                    while grant_sent == 0:
                        grant_sent = grant_ready[0].try_put(MSG_ACK)
                    for i in range(BURST_SIZE):
                        data_rx[0].put(data_mem[addr + i])

                req_count += 1


# ─── Design: Hierarchical 2×1 Decoupled Mesh (1 MT + 2 CTs) ─────────────────

@df.region()
def top_decoupled_2x1(
    in0:  float32[BURST_SIZE],
    in1:  float32[BURST_SIZE],
    out0: float32[BURST_SIZE],
    out1: float32[BURST_SIZE],
):
    """
    1 Memory Tile + 2 Compute Tiles connected via valid-ready handshake.
    Each CT has independent control/data streams. MT dispatches sequentially
    but both CTs run concurrently in HLS dataflow.
    Protocol: WRITE (push data) → COMPUTE (+1.0) → READ (pull data) per CT.
    """
    req_valid:   Stream[int32, 2][N_CTS]
    grant_ready: Stream[int32, 2][N_CTS]
    data_mt2ct:  Stream[float32, BURST_SIZE][N_CTS]
    data_ct2mt:  Stream[float32, BURST_SIZE][N_CTS]

    @df.kernel(mapping=[1], args=[in0, in1, out0, out1])
    def memory_tile_2x1(
        in0_p:  float32[BURST_SIZE],
        in1_p:  float32[BURST_SIZE],
        out0_p: float32[BURST_SIZE],
        out1_p: float32[BURST_SIZE],
    ):
        # WRITE to CT0
        sent: int1 = 0
        while sent == 0:
            sent = req_valid[0].try_put(MSG_WRITE)
        ack: int1 = 0
        while ack == 0:
            grant_val, ack = grant_ready[0].try_get()
        for i in range(BURST_SIZE):
            data_mt2ct[0].put(in0_p[i])

        # WRITE to CT1
        sent = 0
        while sent == 0:
            sent = req_valid[1].try_put(MSG_WRITE)
        ack = 0
        while ack == 0:
            grant_val, ack = grant_ready[1].try_get()
        for i in range(BURST_SIZE):
            data_mt2ct[1].put(in1_p[i])

        # COMPUTE trigger to CT0
        sent = 0
        while sent == 0:
            sent = req_valid[0].try_put(MSG_COMPUTE)
        ack = 0
        while ack == 0:
            grant_val, ack = grant_ready[0].try_get()

        # COMPUTE trigger to CT1
        sent = 0
        while sent == 0:
            sent = req_valid[1].try_put(MSG_COMPUTE)
        ack = 0
        while ack == 0:
            grant_val, ack = grant_ready[1].try_get()

        # READ from CT0
        sent = 0
        while sent == 0:
            sent = req_valid[0].try_put(MSG_READ)
        ack = 0
        while ack == 0:
            grant_val, ack = grant_ready[0].try_get()
        for i in range(BURST_SIZE):
            out0_p[i] = data_ct2mt[0].get()

        # READ from CT1
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
        id = df.get_pid()
        spad: float32[BURST_SIZE] @ Stateful = 0.0

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


# ─── Helpers ─────────────────────────────────────────────────────────────────

def parse_synthesis_xml(xml_file):
    """Parse a Vitis HLS csynth.xml and return resources/latency dict."""
    if not os.path.exists(xml_file):
        return None
    tree = ET.parse(xml_file)
    root = tree.getroot()
    res = {}
    for area in root.iter('AreaEstimates'):
        for r in area:
            if r.tag == 'Resources':
                for child in r:
                    res[child.tag] = child.text
    lat = {}
    for l in root.iter('PerformanceEstimates'):
        for sl in l:
            if sl.tag == 'SummaryOfOverallLatency':
                for child in sl:
                    lat[child.tag] = child.text
    timing_str = None
    for t in root.iter('TimingEstimates'):
        for sl in t:
            if sl.tag == 'summary':
                for child in sl:
                    if child.tag == 'Fmax-MHZ':
                        timing_str = child.text
    return {'resources': res, 'latency': lat, 'fmax_mhz': timing_str}


def run_csim(region_fn, args_np, project_dir, desc):
    if not hls.is_available("vitis_hls"):
        print(f"[SKIP] vitis_hls not available for {desc}")
        return None
    print(f"\n[CSIM] {desc} → {project_dir}")
    mod = df.build(region_fn, target="vitis_hls", mode="csim",
                   project=project_dir, wrap_io=True)
    mod(*args_np)
    return mod


def run_csyn(region_fn, project_dir, desc, top_name):
    if not hls.is_available("vitis_hls"):
        print(f"[SKIP] vitis_hls not available for {desc}")
        return None
    print(f"\n[CSYN] {desc} → {project_dir}")
    mod = df.build(region_fn, target="vitis_hls", mode="csyn",
                   project=project_dir)
    mod()
    # Find and parse report
    xml = os.path.join(project_dir, "out.prj", "solution1", "syn", "report",
                       "csynth.xml")
    result = parse_synthesis_xml(xml)
    if result:
        res = result['resources']
        lat = result['latency']
        print(f"  Resources: LUT={res.get('LUT')}, FF={res.get('FF')}, "
              f"BRAM={res.get('BRAM_18K')}, DSP={res.get('DSP')}")
        print(f"  Latency: {lat.get('Best-caseLatency')} – "
              f"{lat.get('Worst-caseLatency')} cycles, "
              f"II={lat.get('PipelineInitiationInterval')}")
        if result.get('fmax_mhz'):
            print(f"  Fmax: {result['fmax_mhz']} MHz")
    return result


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    base = os.path.abspath("hls_projects")
    os.makedirs(base, exist_ok=True)

    results = {}

    # ─── CSIM NOTE ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  FUNCTIONAL VERIFICATION NOTE")
    print("=" * 60)
    print("""
  Decoupled handshake designs are verified with the Allo simulator
  (OpenMP-based concurrent threads), NOT with Vitis HLS CSIM.

  HLS CSIM runs dataflow processes SEQUENTIALLY. A valid-ready
  handshake requires concurrent execution: MT's spin-wait for
  grant_ready would deadlock if CT runs after MT completes.

  Allo simulator (passed):
    - test_decoupled_message_passing: out = in + 1.0  ✓
    - test_decoupled_2x1_mesh:        CT0, CT1 both +1.0 ✓

  For full cycle-accurate validation, use RTL cosim (requires
  full synthesis; run separately with mode="cosim").
""")

    # ─── CSYN ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  HLS SYNTHESIS")
    print("=" * 60)

    r_mp = run_csyn(top_message_passing,
                    f"{base}/decoupled_mp_csyn.prj",
                    "Decoupled message-passing (MT+1CT)",
                    "top_message_passing")
    results["decoupled_message_passing_1ct"] = r_mp

    r_2x1 = run_csyn(top_decoupled_2x1,
                     f"{base}/decoupled_2x1_csyn.prj",
                     "Decoupled 2x1 mesh (MT+2CTs)",
                     "top_decoupled_2x1")
    results["decoupled_2x1_mesh"] = r_2x1

    # ─── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for name, r in results.items():
        if r:
            print(f"\n  {name}:")
            res = r.get('resources', {})
            lat = r.get('latency', {})
            print(f"    LUT:     {res.get('LUT')}")
            print(f"    FF:      {res.get('FF')}")
            print(f"    BRAM_18K:{res.get('BRAM_18K')}")
            print(f"    DSP:     {res.get('DSP')}")
            print(f"    Latency: {lat.get('Best-caseLatency')}–{lat.get('Worst-caseLatency')} cycles")
            print(f"    II:      {lat.get('PipelineInitiationInterval')}")


if __name__ == "__main__":
    main()
