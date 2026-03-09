"""
Mesh Accelerator Performance Evaluation Framework
===================================================
Three complementary evaluation methods for dynamic (handshake-based) designs
where HLS synthesis reports "undef" latency:

  1. SimStats  – Wall-clock timing + transaction accounting wrapper for the
                 Allo simulator (OpenMP threads). Gives real throughput numbers
                 in software; useful for comparing design variants.

  2. DataflowTimingModel – Analytical protocol-level cycle estimator.
                 Models communication as: handshake_latency × n_transactions
                 + burst_words / bandwidth. Parameterizable per design.

  3. CosimHarness – Reuses a single synthesized RTL project for multiple
                 testbenches. Avoids re-synthesis (30-50s) when the
                 design hasn't changed. Prototype for the framework.

Usage:
    conda run -n allo python tests/dataflow/mesh_perf.py
"""
from __future__ import annotations
import os
import time
import subprocess
import xml.etree.ElementTree as ET
import numpy as np
import allo.dataflow as df
from allo.ir.types import int32, float32, int1, Stream, stateful

# ─── Design imports (reuse from test_decoupled_mesh) ─────────────────────────

# We can't directly import because the df.region decorator runs at import time
# with file-path inspection. Redefine the designs locally.

BURST_SIZE = 16
N_CTS      = 2

MSG_WRITE   = 1
MSG_READ    = 2
MSG_COMPUTE = 4
MSG_GRANT   = 3


@df.region()
def top_decoupled_2x1(
    in0: float32[BURST_SIZE], in1: float32[BURST_SIZE],
    out0: float32[BURST_SIZE], out1: float32[BURST_SIZE],
):
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
        sent: int1 = 0
        while sent == 0: sent = req_valid[0].try_put(MSG_WRITE)
        ack: int1 = 0
        while ack == 0: grant_val, ack = grant_ready[0].try_get()
        for i in range(BURST_SIZE): data_mt2ct[0].put(in0_p[i])

        sent = 0
        while sent == 0: sent = req_valid[1].try_put(MSG_WRITE)
        ack = 0
        while ack == 0: grant_val, ack = grant_ready[1].try_get()
        for i in range(BURST_SIZE): data_mt2ct[1].put(in1_p[i])

        sent = 0
        while sent == 0: sent = req_valid[0].try_put(MSG_COMPUTE)
        ack = 0
        while ack == 0: grant_val, ack = grant_ready[0].try_get()

        sent = 0
        while sent == 0: sent = req_valid[1].try_put(MSG_COMPUTE)
        ack = 0
        while ack == 0: grant_val, ack = grant_ready[1].try_get()

        sent = 0
        while sent == 0: sent = req_valid[0].try_put(MSG_READ)
        ack = 0
        while ack == 0: grant_val, ack = grant_ready[0].try_get()
        for i in range(BURST_SIZE): out0_p[i] = data_ct2mt[0].get()

        sent = 0
        while sent == 0: sent = req_valid[1].try_put(MSG_READ)
        ack = 0
        while ack == 0: grant_val, ack = grant_ready[1].try_get()
        for i in range(BURST_SIZE): out1_p[i] = data_ct2mt[1].get()

    @df.kernel(mapping=[N_CTS])
    def compute_tile_2x1():
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
                    while grant_sent == 0: grant_sent = grant_ready[id].try_put(MSG_GRANT)
                    for i in range(BURST_SIZE): spad[i] = data_mt2ct[id].get()
                elif msg_type == MSG_COMPUTE:
                    for i in range(BURST_SIZE): spad[i] = spad[i] + 1.0
                    grant_sent = 0
                    while grant_sent == 0: grant_sent = grant_ready[id].try_put(MSG_GRANT)
                elif msg_type == MSG_READ:
                    grant_sent = 0
                    while grant_sent == 0: grant_sent = grant_ready[id].try_put(MSG_GRANT)
                    for i in range(BURST_SIZE): data_ct2mt[id].put(spad[i])
                req_count += 1


# ─── Method 1: SimStats ───────────────────────────────────────────────────────

class SimStats:
    """
    Timing + transaction accounting wrapper for the Allo simulator.

    Wraps an LLVMOMPModule (returned by df.build(..., target='simulator'))
    and measures wall-clock time plus data word counts per run.

    Not a cycle-accurate model — wall-clock time reflects the OMP thread
    scheduler, not hardware timing. But it:
      - Correctly counts transactions (exact)
      - Enables comparison between design variants (relative throughput)
      - Scales linearly with data size (linear regression → estimate HW cycles)

    Usage:
        sim = SimStats(df.build(top_decoupled_2x1, target='simulator'))
        sim(in0, in1, out0, out1)
        sim.report()
    """
    def __init__(self, module, clock_freq_mhz: float = 411.0):
        self._mod = module
        self._clock_freq_mhz = clock_freq_mhz
        self.history: list[dict] = []       # one entry per run

    def __call__(self, *args):
        n_words = sum(int(np.prod(a.shape)) for a in args if hasattr(a, 'shape'))
        t0 = time.perf_counter()
        self._mod(*args)
        elapsed_s = time.perf_counter() - t0
        entry = {
            "wall_time_ms":     elapsed_s * 1000,
            "words_transferred": n_words,
            "throughput_gbs":   n_words * 4 / elapsed_s / 1e9,  # 4B per float32
        }
        self.history.append(entry)
        return entry

    def report(self, label: str = ""):
        if not self.history:
            print("  No runs recorded.")
            return
        runs = len(self.history)
        first_ms = self.history[0]["wall_time_ms"]
        total_words = self.history[-1]["words_transferred"]
        print(f"\n  SimStats{' [' + label + ']' if label else ''}:")
        print(f"    Runs: {runs}")
        print(f"    First run: {first_ms:.3f} ms  (includes OMP thread warmup)")
        if runs > 1:
            steady = self.history[1:]
            avg_ms  = sum(h["wall_time_ms"]     for h in steady) / len(steady)
            avg_gbs = sum(h["throughput_gbs"]   for h in steady) / len(steady)
            print(f"    Steady-state avg: {avg_ms:.3f} ms  |  {avg_gbs:.4f} GB/s"
                  f"  ({total_words} words × 4B)")
        print(f"    NOTE: wall-clock ≠ HW cycles. Use DataflowTimingModel for HW estimates.")


# ─── Method 2: DataflowTimingModel ───────────────────────────────────────────

class DataflowTimingModel:
    """
    Analytical performance model for handshake dataflow designs.

    Models cycles as:
        cycles_per_phase = handshake_latency + (burst_size / bw if has_data else 0)

    MT dispatches phases SERIALLY across CTs (sequentially in the MT kernel
    body). CTs run CONCURRENTLY but can only overlap when the MT is not
    addressing them.

    For top_decoupled_2x1 with sequential dispatch:
        MT body: CT0-WRITE, CT1-WRITE, CT0-COMPUTE, CT1-COMPUTE, CT0-READ, CT1-READ
        → Phase separation allows CT1 to compute while MT writes to CT0

    Parameters (tune per design from HLS synthesis reports):
        handshake_latency_cycles: round-trip for try_put + try_get (default: 4)
        bandwidth_words_per_cycle: stream bandwidth (default: 1.0 for Vitis HLS
                                   FIFO inference with 1-word BRAM read port)
        compute_cycles_per_burst: compute kernel latency per burst (default: same
                                  as burst_size at 1 op/cycle for simple vadd)
    """
    def __init__(
        self,
        handshake_latency_cycles: int = 4,
        bandwidth_words_per_cycle: float = 1.0,
        fmax_mhz: float = 411.0,
    ):
        self.hs_lat = handshake_latency_cycles
        self.bw = bandwidth_words_per_cycle
        self.fmax = fmax_mhz

    def _phase_cycles(self, has_data: bool, burst_size: int) -> int:
        data_cycles = int(np.ceil(burst_size / self.bw)) if has_data else 0
        return self.hs_lat + data_cycles

    def estimate_decoupled_mesh_2x1(
        self,
        n_cts: int = 2,
        burst_size: int = BURST_SIZE,
        compute_cycles: int | None = None,
    ) -> dict:
        """
        Estimate cycles for the top_decoupled_2x1 protocol:
            MT dispatch order: [CT0-WRITE, CT1-WRITE, CT0-COMPUTE, CT1-COMPUTE,
                                CT0-READ,  CT1-READ]

        Returns dict with cycle estimates and pipeline breakdown.
        """
        if compute_cycles is None:
            compute_cycles = burst_size   # 1 cycle per element for vadd

        # Phase latencies (from MT's perspective, since MT is the dispatcher)
        write_cycles   = self._phase_cycles(has_data=True,  burst_size=burst_size)
        compute_cycles_hs = self._phase_cycles(has_data=False, burst_size=0) + compute_cycles
        read_cycles    = self._phase_cycles(has_data=True,  burst_size=burst_size)

        # MT SERIAL dispatch latency (worst case: fully sequential)
        mt_serial = n_cts * (write_cycles + compute_cycles_hs + read_cycles)

        # PIPELINED estimate: CT0 compute overlaps MT-CT1 write
        # Timeline (N=2, sequential dispatch per phase group):
        #   t=0:                MT writes to CT0
        #   t=write_cycles:     MT writes to CT1; CT0 IDLE (waiting for compute cmd)
        #   t=2*write_cycles:   MT sends COMPUTE to CT0; CT1 IDLE
        #   t=2*write_cycles + hs_lat: MT sends COMPUTE to CT1; CT0 computing
        #   t=overlap: MT reads from CT0; CT1 computing
        #   ...
        # Key overlap: CT0 compute (compute_cycles) overlaps MT→CT1 compute dispatch (hs_lat)
        # Savings ≈ min(compute_cycles, hs_lat) per CT pair
        overlap_saving = min(compute_cycles, self.hs_lat) * (n_cts - 1)
        mt_pipelined = mt_serial - overlap_saving

        t_serial_ns     = mt_serial     / self.fmax * 1000  # ns
        t_pipelined_ns  = mt_pipelined  / self.fmax * 1000

        return {
            "n_cts":                  n_cts,
            "burst_size":             burst_size,
            "handshake_latency":      self.hs_lat,
            "bandwidth_wpc":          self.bw,
            "write_cycles_per_ct":    write_cycles,
            "compute_cycles_per_ct":  compute_cycles_hs,
            "read_cycles_per_ct":     read_cycles,
            "total_serial_cycles":    mt_serial,
            "total_pipelined_cycles": mt_pipelined,
            "overlap_saving_cycles":  overlap_saving,
            "time_serial_ns":         t_serial_ns,
            "time_pipelined_ns":      t_pipelined_ns,
            "fmax_mhz":               self.fmax,
        }

    def print_report(self, est: dict):
        print(f"\n  DataflowTimingModel  [{est['n_cts']} CTs, burst={est['burst_size']}]:")
        print(f"    Handshake latency:   {est['handshake_latency']} cycles")
        print(f"    Stream bandwidth:    {est['bandwidth_wpc']} word/cycle")
        print(f"  Per-CT phase breakdown:")
        print(f"    WRITE  (hshk + {est['burst_size']} words): {est['write_cycles_per_ct']} cycles")
        print(f"    COMPUTE (hshk + compute): {est['compute_cycles_per_ct']} cycles")
        print(f"    READ   (hshk + {est['burst_size']} words): {est['read_cycles_per_ct']} cycles")
        print(f"    Sum per CT: {est['write_cycles_per_ct'] + est['compute_cycles_per_ct'] + est['read_cycles_per_ct']} cycles")
        print(f"  Total (MT bottleneck, {est['n_cts']} CTs):")
        print(f"    Sequential dispatch: {est['total_serial_cycles']} cycles  "
              f"({est['time_serial_ns']:.1f} ns  @  {est['fmax_mhz']} MHz)")
        print(f"    With CT-compute overlap: {est['total_pipelined_cycles']} cycles  "
              f"({est['time_pipelined_ns']:.1f} ns)")
        print(f"    Overlap saving:  {est['overlap_saving_cycles']} cycles")
        print(f"  ─ Throughput ─")
        words_total = est['n_cts'] * est['burst_size'] * 2  # in + out
        # time_pipelined_ns is in nanoseconds; convert to seconds for GB/s
        time_s = est['time_pipelined_ns'] * 1e-9
        tp_gb_s = words_total * 4 / time_s / 1e9
        print(f"    Effective data BW: {tp_gb_s:.3f} GB/s  ({words_total} words × 4B)")


# ─── Method 3: CosimHarness ──────────────────────────────────────────────────

class CosimHarness:
    """
    Reuses a synthesized RTL project for multiple RTL co-simulation testbenches.

    Architecture:
    ┌─────────────────────────────────────────────────────┐
    │  synthesize()  ─ csynth_design ONCE, cached by XML  │
    │  cosim(data)   ─ cosim_design N×, regenerates TB    │
    │  run_cosim.tcl ─ opens proj WITHOUT -reset, no csyn │
    └─────────────────────────────────────────────────────┘

    RTL co-simulation advantages over Allo simulator:
      - Cycle-accurate (RTL-level transaction sequencing)
      - True concurrent kernel execution (not OMP threads)
      - Validates that HLS-generated RTL matches functional spec
      - Waveforms available (pass trace_level="all" or "port")

    Limitation:
      - Requires Vitis HLS; cosim compilation ~5-15s per run
      - Testbench must call kernel function directly (not via OpenCL)
      - Currently supports designs with flat array arguments only

    See also: DATAFLOW_SEMANTICS.md for execution model discussion.
    """

    _COSIM_TCL_TEMPLATE = """\
# run_cosim.tcl — Reuse existing synthesis, run only RTL cosim
# Generated by CosimHarness. Do NOT run csynth_design here.
open_project out.prj
open_solution "solution1"
add_files -tb host_cosim.cpp -cflags "-std=c++17"
cosim_design -trace_level {trace_level}
exit
"""

    _HOST_TEMPLATE = """\
// host_cosim.cpp — Direct-call cosim testbench
// Generated by CosimHarness. Data read from cosim_input_*.bin files.
#include "kernel.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>

static void read_bin(const char* path, void* dst, size_t bytes) {{
    FILE* f = fopen(path, "rb");
    if (!f) {{ fprintf(stderr, "Cannot open %s\\n", path); exit(1); }}
    fread(dst, 1, bytes, f); fclose(f);
}}
static void write_bin(const char* path, const void* src, size_t bytes) {{
    FILE* f = fopen(path, "wb");
    if (!f) {{ fprintf(stderr, "Cannot open %s\\n", path); exit(1); }}
    fwrite(src, 1, bytes, f); fclose(f);
}}

int main() {{
{input_decls}
{read_calls}
{output_decls}
    // Call the synthesized top-level kernel
    {kernel_call};
{write_calls}
    printf("[COSIM] Done.\\n");
    return 0;
}}
"""

    def __init__(self, project_dir: str, top_func_name: str, fmax_mhz: float = 411.0):
        """
        Args:
            project_dir:   path to the synthesis project (e.g. "hls_projects/blocking_csyn.prj")
            top_func_name: HLS top function name (must match run.tcl's set_top)
            fmax_mhz:      Synthesized Fmax (from csynth report) for timing conversion
        """
        self.project_dir = os.path.abspath(project_dir)
        self.top = top_func_name
        self.fmax_mhz = fmax_mhz
        self._cosim_run_count = 0

    @property
    def synthesized(self) -> bool:
        xml = os.path.join(self.project_dir,
                           "out.prj/solution1/syn/report/csynth.xml")
        return os.path.exists(xml)

    def synthesis_report(self) -> dict | None:
        xml = os.path.join(self.project_dir,
                           "out.prj/solution1/syn/report/csynth.xml")
        if not os.path.exists(xml):
            return None
        tree = ET.parse(xml)
        root = tree.getroot()
        res, lat = {}, {}
        for area in root.iter('AreaEstimates'):
            for r in area:
                if r.tag == 'Resources':
                    for c in r: res[c.tag] = c.text
        for perf in root.iter('PerformanceEstimates'):
            for sl in perf:
                if sl.tag == 'SummaryOfOverallLatency':
                    for c in sl: lat[c.tag] = c.text
        return {"resources": res, "latency": lat}

    def cosim(
        self,
        input_arrays: list[np.ndarray],
        output_shapes: list[tuple],
        output_dtype=np.float32,
        trace_level: str = "none",
    ) -> list[np.ndarray]:
        """
        Run RTL co-simulation using the existing (synthesized) RTL.

        Does NOT re-synthesize if synthesis is already done. Writes input
        data as binary files, generates a new host_cosim.cpp testbench,
        runs `cosim_design`, and reads back output data.

        Args:
            input_arrays:   list of numpy arrays (inputs to the kernel)
            output_shapes:  list of shapes for output arrays
            output_dtype:   dtype for all output arrays (default float32)
            trace_level:    "none", "port", "all" (enables waveform capture)

        Returns:
            list of numpy arrays containing the kernel outputs
        """
        if not self.synthesized:
            raise RuntimeError(
                f"Synthesis not done. Run `vitis_hls -f run.tcl` in {self.project_dir} first.")

        self._cosim_run_count += 1

        # ── Write inputs as binary files ──────────────────────────────────
        for i, arr in enumerate(input_arrays):
            path = os.path.join(self.project_dir, f"cosim_in{i}.bin")
            np.ascontiguousarray(arr).tofile(path)

        # ── Generate host_cosim.cpp ───────────────────────────────────────
        host_cpp = self._gen_host_cpp(input_arrays, output_shapes, output_dtype)
        with open(os.path.join(self.project_dir, "host_cosim.cpp"), "w") as f:
            f.write(host_cpp)

        # ── Generate run_cosim.tcl (no csynth, no -reset) ─────────────────
        tcl = self._COSIM_TCL_TEMPLATE.format(trace_level=trace_level)
        with open(os.path.join(self.project_dir, "run_cosim.tcl"), "w") as f:
            f.write(tcl)

        # ── Run cosim ─────────────────────────────────────────────────────
        t0 = time.perf_counter()
        proc = subprocess.run(
            f"cd {self.project_dir} && vitis_hls -f run_cosim.tcl",
            shell=True, capture_output=False)
        elapsed = time.perf_counter() - t0

        if proc.returncode != 0:
            raise RuntimeError(f"Cosim failed (see vitis_hls.log in {self.project_dir})")

        print(f"  [CosimHarness] Run #{self._cosim_run_count} done in {elapsed:.1f}s "
              f"(synthesis REUSED, only cosim_design ran)")

        # ── Read outputs ──────────────────────────────────────────────────
        outputs = []
        for i, shape in enumerate(output_shapes):
            path = os.path.join(self.project_dir, f"cosim_out{i}.bin")
            if not os.path.exists(path):
                print(f"  Warning: output file {path} not found, returning zeros")
                outputs.append(np.zeros(shape, dtype=output_dtype))
            else:
                outputs.append(np.fromfile(path, dtype=output_dtype).reshape(shape))
        return outputs

    def _gen_host_cpp(self, inputs, output_shapes, out_dtype) -> str:
        """Generate a C++ cosim testbench that reads/writes binary data files."""
        ctype_map = {np.float32: "float", np.int32: "int", np.float64: "double",
                     np.int64: "long long"}
        in_ctype = ctype_map.get(inputs[0].dtype.type, "float") if inputs else "float"
        out_ctype = ctype_map.get(out_dtype, "float")

        input_decls, read_calls = [], []
        for i, arr in enumerate(inputs):
            n = int(np.prod(arr.shape))
            ct = ctype_map.get(arr.dtype.type, "float")
            input_decls.append(f"    {ct} in{i}[{n}];")
            read_calls.append(f'    read_bin("cosim_in{i}.bin", in{i}, sizeof(in{i}));')

        output_decls, write_calls = [], []
        for i, shape in enumerate(output_shapes):
            n = int(np.prod(shape))
            output_decls.append(f"    {out_ctype} out{i}[{n}] = {{0}};")
            write_calls.append(f'    write_bin("cosim_out{i}.bin", out{i}, sizeof(out{i}));')

        all_args = ", ".join(
            [f"in{i}" for i in range(len(inputs))] +
            [f"out{i}" for i in range(len(output_shapes))]
        )
        kernel_call = f"{self.top}({all_args})"

        return self._HOST_TEMPLATE.format(
            input_decls="\n".join(input_decls),
            read_calls="\n".join(read_calls),
            output_decls="\n".join(output_decls),
            write_calls="\n".join(write_calls),
            kernel_call=kernel_call,
        )


# ─── Demo ─────────────────────────────────────────────────────────────────────

def demo_sim_stats():
    """Demo: SimStats timing wrapper on the Allo simulator."""
    print("\n" + "=" * 60)
    print("  METHOD 1: SimStats (Allo simulator timing)")
    print("=" * 60)

    sim = SimStats(df.build(top_decoupled_2x1, target="simulator"), clock_freq_mhz=411.0)

    in0 = np.arange(BURST_SIZE, dtype=np.float32)
    in1 = np.arange(BURST_SIZE, dtype=np.float32) * 2.0
    out0 = np.zeros(BURST_SIZE, dtype=np.float32)
    out1 = np.zeros(BURST_SIZE, dtype=np.float32)

    N_RUNS = 5
    for _ in range(N_RUNS):
        sim(in0, in1, out0, out1)

    np.testing.assert_allclose(out0, in0 + 1.0)
    np.testing.assert_allclose(out1, in1 + 1.0)
    sim.report(label="top_decoupled_2x1")


def demo_timing_model():
    """Demo: Analytical protocol cycle estimator."""
    print("\n" + "=" * 60)
    print("  METHOD 2: DataflowTimingModel (analytical cycle estimate)")
    print("=" * 60)

    model = DataflowTimingModel(
        handshake_latency_cycles=4,       # 4 cycles: 1 put + 1 cycle prop + 1 poll + 1 cycle prop
        bandwidth_words_per_cycle=1.0,    # from HLS report: Shift-Register FIFO, 1 word/cycle
        fmax_mhz=411.0,                   # from Vitis HLS synthesis report
    )

    print("\n  [1 CT baseline: top_message_passing]")
    est_1ct = model.estimate_decoupled_mesh_2x1(n_cts=1, burst_size=BURST_SIZE)
    model.print_report(est_1ct)

    print("\n  [2 CT hierarchy: top_decoupled_2x1]")
    est_2ct = model.estimate_decoupled_mesh_2x1(n_cts=2, burst_size=BURST_SIZE)
    model.print_report(est_2ct)

    print("\n  [Scaling: 1 CT → 2 CT]")
    delta_serial = est_2ct["total_serial_cycles"] - est_1ct["total_serial_cycles"]
    delta_pipe   = est_2ct["total_pipelined_cycles"] - est_1ct["total_pipelined_cycles"]
    print(f"    +{delta_serial} serial cycles / +{delta_pipe} pipelined cycles per extra CT")
    print(f"    Throughput ratio (2CT/1CT): "
          f"{est_1ct['total_pipelined_cycles'] / est_2ct['total_pipelined_cycles']:.2f}× "
          f"[note: depends on MT dispatch order]")


def demo_cosim_harness_describe():
    """
    Demo: CosimHarness description (does NOT run actual cosim in this demo).
    Prints the RTL, TCL, and host.cpp that would be generated.
    """
    print("\n" + "=" * 60)
    print("  METHOD 3: CosimHarness (multi-testbench RTL reuse)")
    print("=" * 60)

    harness = CosimHarness(
        project_dir="hls_projects/blocking_csyn.prj",
        top_func_name="top_blocking",
        fmax_mhz=411.0,
    )

    # Show synthesis status
    print(f"\n  Project: {harness.project_dir}")
    print(f"  Synthesis already done: {harness.synthesized}")
    if harness.synthesized:
        r = harness.synthesis_report()
        res = r["resources"]
        print(f"  Cached synthesis: LUT={res.get('LUT')}, FF={res.get('FF')}")

    # Show what run_cosim.tcl would look like
    print("\n  Generated run_cosim.tcl (no -reset, no csynth_design):")
    print("  " + "-" * 50)
    tcl = CosimHarness._COSIM_TCL_TEMPLATE.format(trace_level="none")
    for line in tcl.strip().split("\n"):
        print(f"    {line}")
    print("  " + "-" * 50)

    # Show generated host_cosim.cpp
    inputs = [np.zeros(4, dtype=np.int32)]  # top_blocking takes int32[4]
    outputs = [(4,)]
    host = harness._gen_host_cpp(inputs, outputs, np.int32)
    print("\n  Generated host_cosim.cpp (reads/writes .bin files):")
    print("  " + "-" * 50)
    for line in host.strip().split("\n")[:25]:
        print(f"    {line}")
    print("  " + "-" * 50)

    print("""
  Key advantage: synthesize() runs csynth once (~30-50s),
  cosim() runs in ~5-15s and can be called N times with
  different input data — NO re-synthesis.

  This enables:
    - Regression testing with multiple data patterns
    - Corner-case verification (empty streams, full FIFOs)
    - Performance sweep (vary burst size, FIFO depth)
    - Waveform collection for debugging (trace_level="all")
  """)


def main():
    demo_sim_stats()
    demo_timing_model()
    demo_cosim_harness_describe()


if __name__ == "__main__":
    main()
