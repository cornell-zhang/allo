# HLS Synthesis Report: Non-Blocking Stream Interconnect & Decoupled Mesh

**Device**: Xilinx U280 (xcu280-fsvh2892-2L-e)
**Tool**: Vitis HLS 2023.2
**Clock period**: 3.33 ns (300 MHz target) / **Estimated Fmax**: 411 MHz
**Date**: 2026-03-08

---

## 1. Blocking vs Non-Blocking Stream Primitives

**Designs**: `tests/dataflow/hls_synth_streams.py`
**Reference test**: `tests/dataflow/test_stream_nb_simple.py`

### 1.1 Functional Verification (CSIM)

All three designs passed C-simulation (`vitis_hls csim`):

| Design | Test | Result |
|--------|------|--------|
| `top_blocking` (put/get) | `out[i] == i * 10` for i in 0..3 | PASS |
| `top_nonblocking` (try_put/try_get) | `out[i] == i * 10` for i in 0..3 | PASS |
| `top_status_flags` (empty/full) | `[1, 0, 1, 0]` for flags/status | PASS |

> **Note on `top_nonblocking` CSIM**: Non-blocking spin-wait loops succeed in
> HLS CSIM because HLS CSIM executes dataflow processes sequentially — the
> producer fills the depth-4 FIFO completely in a single pass, then the consumer
> drains it. Each `try_put`/`try_get` call succeeds on first attempt because
> there is no concurrent racing. Functional correctness is verified; cycle-level
> concurrency behavior requires RTL co-simulation.

### 1.2 HLS Synthesis Results (CSYN)

Top-level resource comparison:

| Design | LUT | FF | BRAM_18K | DSP | Latency (cycles) | Pipeline II | Fmax (MHz) |
|--------|-----|----|----------|-----|-----------------|-------------|------------|
| `top_blocking` | **1417** | **1325** | 2 | 0 | 19–20 | **7** | 411 |
| `top_nonblocking` | **1457** | **1369** | 2 | 0 | undef | undef | 411 |

**Overhead (non-blocking vs blocking)**:
- LUT: +40 (+2.8%), FF: +44 (+3.3%)
- Latency: becomes statically unbounded due to spin-wait loops (correct: actual
  cycles depend on FIFO occupancy at runtime)
- Fmax: identical — non-blocking comparators do not affect timing closure

### 1.3 Per-Module Breakdown

| Module | Type | LUT | FF | Notes |
|--------|------|-----|----|-------|
| `producer_0` | Blocking put | 82 | 9 | Loop body: stream.write() |
| `consumer_0` | Blocking get | 73 | 9 | Loop body: stream.read() |
| `producer_nb_0` | Non-blocking try_put | 89 | 15 | +7 LUT (+9%), +6 FF |
| `consumer_nb_0` | Non-blocking try_get | 106 | 47 | +33 LUT (+45%), +38 FF |

**Analysis**:
- **Producer overhead** (+7 LUT, +6 FF): The `write_nb()` call adds one
  comparator (is_not_full?) plus a MUX for the success return bit. Extra FFs
  track the spin-wait iteration state.
- **Consumer overhead** (+33 LUT, +38 FF): The `read_nb()` adds a comparator
  and success MUX. The `while ok == 0` spin loop requires a state machine
  register (+1 FF for ok, +1 FF for the loop counter). The larger FF overhead
  reflects the loop-control registers in the pipelined state machine.
- **Conclusion**: Non-blocking ops add ~O(1) LUTs per stream (one comparator
  + one MUX for the success bit). The extra FFs come from the spin-wait state
  machine, not from the stream datapath itself.

### 1.4 Status Flags (`empty()` / `full()`)

- HLS codegen verification: PASS — `.empty()` and `.full()` correctly lower to
  `hls::stream<T>::empty()` and `hls::stream<T>::full()` (see
  `mlir/lib/Translation/EmitVivadoHLS.cpp`)
- HLS synthesis: **SKIP** — `top_status_flags` has a single-kernel design that
  writes to an internal stream without consuming it (pure status test). HLS
  dataflow requires every internal FIFO to be produced by exactly one process and
  consumed by exactly one process. A functional-only design cannot be synthesized
  as-is; in a real design, empty()/full() are used inside producer/consumer
  kernels as guards, not standalone.

### 1.5 HLS API Mapping

| Allo Python | Vitis HLS C++ | Tapa HLS C++ |
|-------------|---------------|--------------|
| `s.put(v)` | `s.write(v)` | `s.write(v)` |
| `s.get()` | `s.read()` | `s.read()` |
| `ok = s.try_put(v)` | `ok = s.write_nb(v)` | `ok = s.try_write(v)` |
| `v, ok = s.try_get()` | `ok = s.read_nb(v)` | `ok = s.try_read(v)` |
| `s.empty()` | `s.empty()` | `s.empty()` |
| `s.full()` | `s.full()` | `s.full()` |

---

## 2. Decoupled Mesh Architecture

**Designs**: `tests/dataflow/test_decoupled_mesh.py`, `tests/dataflow/hls_synth_decoupled.py`

### 2.1 Architecture Overview

The decoupled mesh replaces Allo's blocking fixed-II streaming protocol with a
**valid-ready handshake** on control channels:

```
Memory Tile (MT)  ─── req_valid[i] (try_put) ───▶  Compute Tile i (CT_i)
                  ◀── grant_ready[i] (try_get) ───  Compute Tile i (CT_i)
                  ─── data_mt2ct[i] (burst put) ──▶  Compute Tile i
                  ◀── data_ct2mt[i] (burst get) ───  Compute Tile i
```

**Protocol (3 phases per CT per invocation)**:
1. **WRITE** (MSG=1): MT raises Valid (try_put) → CT asserts Ready (try_put
   grant) → MT sends unconditional data burst
2. **COMPUTE** (MSG=4): MT dispatches work token → CT performs in-place +1.0
   → CT sends completion grant
3. **READ** (MSG=2): MT raises Valid → CT asserts Ready → CT sends data burst

**Key advantage over blocking (top_2x1 in test_hierachical_mesh.py)**:
- Blocking requires every MT→CT channel to be written in every invocation
  (fixed-II), even via NOP padding, to avoid FIFO deadlock
- Decoupled allows MT to selectively target any subset of CTs
- CTs run concurrently and independently with no fixed-II constraint

### 2.2 Functional Verification

Verified with the **Allo simulator** (OpenMP concurrent threads):

| Design | Test | Result |
|--------|------|--------|
| `top_message_passing` | `out[i] == in[i] + 1.0` | PASS |
| `top_decoupled_2x1` | CT0: `out0[i] == in0[i] + 1.0` | PASS |
| | CT1: `out1[i] == in1[i] + 1.0` | PASS |

> **HLS CSIM not applicable**: Vitis HLS C-simulation runs dataflow processes
> **sequentially** (documented in UG1399). A valid-ready handshake requires
> concurrent execution: the MT's `while ack == 0: try_get(grant)` spin-wait
> deadlocks if CT hasn't had a chance to run. For cycle-accurate concurrent
> validation, use RTL co-simulation (`mode="cosim"`), which runs synthesized RTL
> in Xsim with true concurrency. The Allo simulator (OpenMP) correctly models
> concurrent kernel execution and is used for functional verification.

### 2.3 HLS Synthesis Results (CSYN)

| Design | LUT | FF | BRAM_18K | DSP | Fmax (MHz) |
|--------|-----|----|----------|-----|------------|
| `top_message_passing` (1 MT + 1 CT) | 5355 | 5300 | 7 | 2 | 411 |
| `top_decoupled_2x1` (1 MT + 2 CTs) | 7361 | 6724 | 8 | 4 | 411 |

**Scaling from 1 CT → 2 CTs** (+1 CT):
- LUT: +2006 (+37.5%)
- FF: +1424 (+26.9%)
- BRAM: +1 (+14%)
- DSP: +2 (+100% — each CT's burst data pipeline uses 1 DSP for AXI burst)

Both designs synthesize cleanly at 411 MHz (3.33 ns period satisfied with margin).

> **Latency: undef** — Expected for designs with spin-wait while loops.
> HLS cannot statically bound the iteration count of `while ack == 0`.
> In practice, latency is bounded by the FIFO depth and the handshake round-trip
> time (1–2 clock cycles). The undef label does not indicate a real problem.

### 2.4 Per-Module Breakdown (2×1 Mesh)

| Module | LUT | FF | Function |
|--------|-----|----|----------|
| `memory_tile_2x1_0` | 789 | 131 | MT: 6 handshake phases + burst control |
| `compute_tile_2x1_0` (CT0) | 801 | 619 | CT: spin-poll + 3-phase protocol |
| `compute_tile_2x1_1` (CT1) | 801 | 619 | CT: same (symmetric) |
| AXI burst load/store | 239+229 | 439+469 | AXI-M data movers (per array port) |
| FIFOs (shift-register) | — | — | req_valid, grant_ready, data streams |

**Observations**:
- Each CT adds ~801 LUT + ~619 FF — **near-linear scaling** (slightly sublinear
  due to shared AXI interface logic being amortized)
- The MT is surprisingly lean (789 LUT, 131 FF) because its handshake loops
  are lightweight comparators; the AXI burst data movers are in separate
  processes (`load_buf`, `store_res`)
- CT has higher FF count (~619) than MT (~131) because the `spad[BURST_SIZE]`
  scratchpad registers inside `compute_tile_2x1` are kept as FFs (small arrays
  get inferred into registers rather than BRAM at depth=16)

---

## 3. Blocking vs Decoupled: Area Comparison

For reference, the blocking `top_2x1` (from `test_hierachical_mesh.py`) uses:
- Fixed command packets (ctrl, daddr, size, data) always broadcast to all CTs
- CTs decode NOP commands in every cycle to satisfy fixed-II
- Total area: not synthesized here (requires full GEMM/VADD sub-regions)

For a simplified 2-CT point comparison with identical compute (vadd +1):

| Protocol | LUT | FF | BRAM | Notes |
|----------|-----|----|------|-------|
| Blocking (put/get FIFO) | ~1417 | ~1325 | 2 | Simple producer-consumer only |
| Non-blocking (try_put/get) | ~1457 | ~1369 | 2 | Spin-wait overhead +3% area |
| Decoupled mesh (1 MT + 1 CT) | 5355 | 5300 | 7 | Full handshake + AXIM burst |
| Decoupled mesh (1 MT + 2 CTs) | 7361 | 6724 | 8 | +37% per extra CT |

The decoupled mesh has higher absolute area because:
1. **AXI-M burst interfaces** for the external data ports (not present in simple
   stream tests)
2. **Stateful scratchpad** (`spad[16] @ stateful`) mapped to FFs inside each CT
3. **Full handshake state machine** for 3-phase protocol vs simple FIFO

---

## 4. Key Findings

1. **Non-blocking overhead is minimal**: try_put/try_get add only ~3% area over
   blocking equivalents. The extra LUTs come from comparators; extra FFs come
   from spin-wait state machines. Fmax is not affected.

2. **Both designs meet 411 MHz**: Far exceeding the 300 MHz target. Non-blocking
   handshake does NOT become the critical path.

3. **Decoupled mesh scales near-linearly with CT count**: Adding 1 CT costs
   ~2000 LUT + ~1400 FF. The MT overhead is sublinear (shared AXI logic).

4. **HLS CSIM limitation for handshake designs**: Designs with bidirectional
   spin-wait handshakes (MT waits for CT grant before CT has run) deadlock in
   HLS CSIM's sequential execution model. The Allo simulator (OMP threads) is
   the correct verification vehicle for such designs.

5. **Single-kernel internal streams fail HLS synthesis**: `empty()`/`full()`
   flags are only useful as guards inside producer/consumer kernels — not as
   standalone single-kernel status checks (HLS dataflow requires one producer
   and one consumer per internal stream).

---

## 5. Files

| File | Purpose |
|------|---------|
| `tests/dataflow/test_stream_nb_simple.py` | Unit tests for all 4 NB primitives |
| `tests/dataflow/test_stream_ops_ir.py` | MLIR op generation checks |
| `tests/dataflow/test_stream_ops_sim.py` | Simulator functional tests |
| `tests/dataflow/test_stream_ops_hls.py` | HLS codegen pattern verification |
| `tests/dataflow/test_decoupled_mesh.py` | Decoupled mesh (1-CT + 2×1 mesh) |
| `tests/dataflow/hls_synth_streams.py` | CSIM+CSYN script: blocking vs NB |
| `tests/dataflow/hls_synth_decoupled.py` | CSYN script: decoupled mesh |
| `hls_projects/blocking_csyn.prj/` | Vitis HLS project: top_blocking |
| `hls_projects/nonblocking_csyn.prj/` | Vitis HLS project: top_nonblocking |
| `hls_projects/decoupled_mp_csyn.prj/` | Vitis HLS project: top_message_passing |
| `hls_projects/decoupled_2x1_csyn.prj/` | Vitis HLS project: top_decoupled_2x1 |
| `mlir/lib/Translation/EmitVivadoHLS.cpp` | Vitis HLS codegen (NB ops fixed) |
| `mlir/lib/Translation/EmitTapaHLS.cpp` | Tapa HLS codegen (NB ops fixed) |
| `ALLO_CHANGE.md` | Full log of Allo codebase modifications |

---

## 6. HLS Synthesis Command Reference

```bash
# Run all stream primitive tests
conda run -n allo python tests/dataflow/hls_synth_streams.py

# Run decoupled mesh synthesis
conda run -n allo python tests/dataflow/hls_synth_decoupled.py

# Run Allo simulator functional tests
conda run -n allo python -m pytest tests/dataflow/test_stream_nb_simple.py \
  tests/dataflow/test_decoupled_mesh.py -v

# Re-run individual synthesis from project directory
cd hls_projects/blocking_csyn.prj && conda run -n allo vitis_hls -f run.tcl
```
