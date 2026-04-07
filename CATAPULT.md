# Catapult HLS Synthesis: Findings and Analysis

---

## 0. Design Visualization

### 0.1 Structural Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          top_decoupled_2x1                                  │
│                                                                             │
│  External Memory                                                            │
│  ┌──────────────────┐      ┌────────────────────────────┐                  │
│  │ v0.d [A] 512-bit ├─────►│                            │                  │
│  │ v1.d [B] 512-bit ├─────►│   memory_tile_2x1_0 (MT)  │                  │
│  │ v2.d [out0]      │◄─────┤    Lat=67 / Thru=69 cyc   │                  │
│  │ v3.d [out1]      │◄─────┤    Datapath:  2,971 score  │                  │
│  └──────────────────┘      │    Register: 13,209 score  │                  │
│                             └──────┬──────────────┬──────┘                  │
│                                    │              │                         │
│          ┌─────────────────────────┘              └──────────────────────┐  │
│          │  ←── 4 channel types per CT ────────────────────────────────► │  │
│          │                                                                │  │
│    Data  │ ████ depth=16, 32-bit, area=4279 (v184:cns)                  │  │
│    Ctrl  │ ▓▓▓  depth= 3, 32-bit, area= 862 (v186:cns)                  │  │
│  Result  │ ░░░  depth= 2, 32-bit, area= 594 (v188:cns)                  │  │
│   Token  │ ·    depth= 1, 32-bit, area= 326 (v191:cns)                  │  │
│          │                                                                │  │
│          ▼                                                                ▼  │
│  ┌──────────────────────┐              ┌──────────────────────┐            │
│  │ compute_tile_2x1_0   │              │ compute_tile_2x1_1   │            │
│  │       (CT0)          │              │       (CT1)          │            │
│  │  spad[16] FP32       │              │  spad[16] FP32       │            │
│  │  Lat=295 / Thru=298  │              │  Lat=295 / Thru=298  │            │
│  │  Datapath:  4,247    │              │  Datapath:  4,247    │            │
│  │  Register: 10,744    │              │  Register: 10,744    │            │
│  └──────────────────────┘              └──────────────────────┘            │
└─────────────────────────────────────────────────────────────────────────────┘

 8 FIFOs total (4 types × 2 CTs):
   Data  ×2: MT→CT0, MT→CT1          4,279 each → total  8,558 (69% of interconnect)
   Ctrl  ×2: MT↔CT0, MT↔CT1            862 each → total  1,725
   Result×2: CT0→MT, CT1→MT             594 each → total  1,188
   Token ×2: CT↔MT (done/ack)        594 + 326  → total    920
   ─────────────────────────────────────────────────────────────
   Total interconnect:                               12,391 (19% of design area)
```

---

### 0.2 Execution Timeline

Based on the Loop Execution Profile table in `cycle.rpt`.
1 invocation = 298 cycles = 596 ns @ 500 MHz (`Thru=298`).

```
Cycle:  0        32       64  69   99      198       297 298
        |         |        |   |    |        |         |   |

MT:     [══A×16══][══B×16══][5]
        l_S_i_2_i  l_S_i_12_i2           (32 cyc each, 45.7% each)
        └── done at cycle 69 ────────────────────────────────

CT0:    [3][──RX──][────Compute (48)────][──TX──] ← while iter 1 (99 cyc)
             16                 48          32
        [3][──RX──][────Compute────────][──TX──]   ← while iter 2
        [3][──RX──][────Compute────────][──TX──]   ← while iter 3
                                                    298 cyc total

CT1:    (identical to CT0, runs in parallel)

Legend:
  [3]         = while handshake (try_get grant token)
  [──RX──]    = l_S_i_1: receive 16 FP32 from data FIFO (16 cyc, 1 c-step/elem)
  [──Compute] = l_S_i_2: FP accumulate 16 elements  (48 cyc, 3 c-steps/elem)
  [──TX──]    = l_S_i_5: send 16 FP32 to result FIFO (32 cyc, 2 c-steps/elem)
```

---

### 0.3 Concurrency Analysis

#### MT ↔ CT: Pipeline Overlap

**MT (69 cyc) and CTs (298 cyc) run concurrently.** The Data FIFOs (depth=16) absorb the producer-consumer speed mismatch.

- MT `l_S_i_2_i` (32 cyc) and CT `l_S_i_1` RX (16 cyc) **overlap via the FIFO**: as soon as MT writes the first element, the CT begins reading it (streaming overlap from cycle ~1).
- MT finishes at cycle 69; CTs run until cycle 298. **Cycles 70–298: CTs run alone** — this window covers most of Compute (144 cyc) and TX (96 cyc).
- Area perspective: Data FIFO depth=16 is needed to buffer the 4.3× speed ratio (298/69). Catapult's automatically inferred depth=16 is a full-burst buffering choice, accepting the area cost (4,279 each, 69% of total FIFO area) to avoid stalls.

#### CT0 ↔ CT1: Full Parallelism

**CT0 and CT1 run the same 298-cycle schedule in parallel.**
- `cycle.rpt`: both modules report `Latency=295, Throughput=298` with identical loop structure.
- MT broadcasts the same data to CT0 and CT1 via independent FIFOs, so there is no data dependency between the two CTs.
- The reported design total latency of 657 cycles is a sequential sum; the **actual critical path when run concurrently is 298 cycles** (CT-dominated).

#### Within a CT: RX → Compute → TX is Sequential (No Intra-Tile Pipeline)

Per `while` iteration (99 cycles):

| Phase | Cycles | Overlap with other modules |
|-------|--------|---------------------------|
| Handshake (Ctrl) | 3 | — |
| RX (`l_S_i_1`) | 16 | Overlaps with MT `l_S_i_2_i` writes via FIFO |
| Compute (`l_S_i_2`) | 48 | MT already done or sending next burst |
| TX (`l_S_i_5`) | 32 | Overlaps with MT result-receive loop via FIFO |

RX → Compute → TX are **sequential within a tile** (no HLS pipeline pragma applied).
Future optimization: applying double-buffering to `l_S_i_1` (RX) and `l_S_i_2` (Compute) would overlap them, theoretically reducing CT throughput by up to 48+32 = 80 cycles.

---

## 1. Design Overview

**Design:** `top_decoupled_2x1` — Decoupled 2×1 mesh accelerator (1 Memory Tile + 2 Compute Tiles)
**Source:** `tests/dataflow/test_decoupled_mesh.py` → `catapult_decoupled_2x1.prj/kernel.cpp`
**Architecture:** Valid-ready handshake between MT and CTs via `ac_channel` FIFOs; MT streams matrix rows to CTs, CTs compute FP dot-products and stream results back.

### Constants

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `M, N, K` | 2 | Matrix dimension |
| `P0, P1`  | 4 | Partition factors |
| `MEM_SIZE` | 256 | External memory words |
| `IMEM_SIZE` | 8 | Instruction/config memory |
| `BW`      | 16 | Bit-width per element (effective: 32-bit FP used) |
| `TILE_M`  | 2 | Tiles in M dimension |
| Elements/stream | 16 | Payload size per streaming burst |

---

## 2. Synthesis Setup

| Parameter | Value |
|-----------|-------|
| Tool | Catapult Ultra 2024.2 (build 1130128) |
| Target library | `nangate-45nm_beh` |
| Clock | `clk`, rising edge, 2.0 ns period (500 MHz) |
| Clock uncertainty | 0% |
| Synthesis mode | **Block synthesis** (`solution design set <fn> -block` for each tile) |
| Reported timing slack | 0.041 ns (timing closure met) |
| Max delay | 1.959 ns |

### TCL Flow (catapult_decoupled_2x1.prj/run.tcl)

```tcl
solution options set /Input/CppStandard c++11
solution options set /Input/CompilerFlags {{-D_GLIBCXX_USE_CXX11_ABI=0}}
solution file add kernel.cpp -type C++
directive set -DESIGN_HIERARCHY top_decoupled_2x1
directive set -CLOCKS {clk {-CLOCK_PERIOD 2.0}}
solution options set /Output/OutputVerilog true
solution library add nangate-45nm_beh

go analyze
solution design set memory_tile_2x1_0 -block
solution design set compute_tile_2x1_0 -block
solution design set compute_tile_2x1_1 -block

go compile
solution library add ccs_sample_mem
go assembly
go extract
```

---

## 3. Latency Results

### 3.1 Summary

| Module | Latency (cycles) | Throughput (cycles) | Time @ 500 MHz |
|--------|-----------------|---------------------|----------------|
| `compute_tile_2x1_0` | 295 | 298 | 596 ns |
| `compute_tile_2x1_1` | 295 | 298 | 596 ns |
| `memory_tile_2x1_0`  | 67  | 69  | 138 ns |
| **Design Total** | **657** (sequential) | **298** (CTs dominate) | **596 ns** |

Tiles run **concurrently**: MT sends data in 69 cycles while CTs compute in parallel.
Critical path: CT throughput (298 cycles) determines overall system throughput.

---

### 3.2 Compute Tile (CT) — Latency Decomposition

Each CT processes 16 FP elements from MT, accumulates a dot-product, and streams the result back. The outer `while` loop runs **3 handshake rounds** (matching the data/token protocol).

#### Per-Iteration Loop Breakdown (single `while` iteration = 99 cycles)

| Loop | Role | Iters | C-steps/iter | Cycles/iter |
|------|------|-------|--------------|-------------|
| `while` header | Handshake control (try_get grant) | — | 3 | 3 |
| `l_S_i_1` | **Receive**: read 16 FP elements from data stream | 16 | 1 | 16 |
| `l_S_i_2` | **Compute**: FP add/accumulate 16 elements | 16 | 3 | 48 |
| `l_S_i_5` | **Send**: write 16 FP results to result stream | 16 | 2 | 32 |
| **Total per iter** | | | | **99** |

#### Full CT Execution (3 `while` iterations = 297 cycles + 1 overhead = 298)

| Category | Loop(s) | Cycles | % of Throughput | Notes |
|----------|---------|--------|-----------------|-------|
| **Communication (RX)** | `l_S_i_1` × 3 | **48** | 16.1% | Stream-read 16 FP32 from MT data FIFO |
| **Compute (FP)** | `l_S_i_2` × 3 | **144** | 48.3% | FP accumulate: 3 c-steps/element (FP add latency) |
| **Communication (TX)** | `l_S_i_5` × 3 | **96** | 32.2% | Stream-write 16 FP32 to result FIFO |
| **Control (handshake)** | `while` header × 3 | **9** | 3.0% | `try_get`/`try_put` on grant tokens |
| **Overhead** | `main` | **1** | 0.3% | Loop prologue |
| **Total** | | **298** | 100% | Throughput (II) |

**Key insight**: FP compute dominates at 48%. Communication accounts for 48% total (RX + TX),
confirming the design is **compute-memory balanced** at this datapath width.

---

### 3.3 Memory Tile (MT) — Latency Decomposition

MT reads two 16-element arrays from external memory (`v0.d` = matrix A, `v1.d` = matrix B) and streams each to both CTs simultaneously. It simultaneously receives result streams from CTs and writes them to external memory (`v2.d`, `v3.d`).

#### MT Loop Breakdown

| Loop | Role | Iters | C-steps/iter | Total Cycles | % |
|------|------|-------|--------------|--------------|---|
| `l_S_i_2_i` | **Storage→Comm**: read A + stream to CT0/CT1 | 16 | 2 | **32** | 45.7% |
| `l_S_i_12_i2` | **Storage→Comm**: read B + stream to CT0/CT1 | 16 | 2 | **32** | 45.7% |
| `main` | Overhead + control token management | — | 5 | **5** | 7.1% |
| `core:rlp` | Ring-loop reset overhead | — | 1 | **1** | 1.4% |
| **Total** | | | | **69** → 70 | 100% |

#### MT Subcategory Breakdown

| Category | Cycles | % | Notes |
|----------|--------|---|-------|
| **Storage→Comm (read A + send)** | **32** | 46% | 16 reads × 2 c-steps (mem read + channel write fused) |
| **Storage→Comm (read B + send)** | **32** | 46% | Same pattern for B matrix |
| **Control + overhead** | **6** | 9% | Token generation, output memory write control, prologue |
| **Total** | **69** | 100% | |

**Key insight**: MT is almost entirely I/O-bound. The 2-cycle per element cost reflects the
external memory read (1 cycle) + FIFO broadcast write to 2 CTs (1 cycle). MT is **not**
a bottleneck — it completes in 69 cycles vs. CT's 298 cycles.

---

## 4. Area Results

### 4.1 Design-Level Area Summary

*(Catapult area score units — scheduling metric, not physical nm²)*

| Category | Post-Assignment Score | % |
|----------|-----------------------|---|
| REG (pipeline registers, arrays) | 35,029 | 54% |
| FUNC (datapath logic: adders, MUL) | 14,551 | 22% |
| MUX (data path multiplexers) | 14,313 | 22% |
| LOGIC (control logic) | 1,408 | 2% |
| **Total (excl. I/O + FIFOs)** | **65,301** | |
| FSM (finite state machine regs) | 161 | — |
| **Design Total post-assignment** | **65,462** | |

#### Per-Module Area (CRAAS-12 final schedule)

| Module | Datapath | Register | Total |
|--------|----------|----------|-------|
| `compute_tile_2x1_0` | 4,246.68 | 10,744.27 | **14,990.96** |
| `compute_tile_2x1_1` | 4,246.68 | 10,744.27 | **14,990.96** |
| `memory_tile_2x1_0` | 2,971.23 | 13,208.50 | **16,179.73** |
| Sum of tiles | 11,464.59 | 34,697.04 | **46,161.65** |
| Interconnect FIFOs (top-level) | — | — | **12,390.81** |
| Other (lib primitives, FSM) | — | — | **~910** |
| **Design Total** | | | **~65,462** |

---

### 4.2 Compute Tile (CT) — Area Decomposition

Each CT contains FP arithmetic units and a scratchpad register array (`spad[16]`).
FIFOs are instantiated at the top level and not counted in tile area.

| Subcategory | Score | % of CT | What it contains |
|-------------|-------|---------|------------------|
| **Compute** (Datapath) | **4,246.68** | 28% | FP adder/accumulator (3 c-step FP add); loop counter logic; address arithmetic for spad indexing; MUX for operand selection |
| **Storage** (Register) | **10,744.27** | 72% | `spad[16]`: 16 × 32-bit FP scratchpad (512 bits); accumulator register; pipeline hold registers for stream I/O; loop induction variables |
| **Communication** | **0** | — | Channels (FIFOs) are top-level; `ccs_in_wait`/`ccs_out_wait` I/O ports are 0-area interface wrappers |
| **CT Total** | **14,990.96** | 100% | |

**Observation**: Storage dominates at 72%, driven by the 16-element FP scratchpad.
For larger tile sizes, storage cost would grow linearly with scratchpad depth.

---

### 4.3 Memory Tile (MT) — Area Decomposition

MT manages 4 external 512-bit memory interfaces (v0..v3, each 16×32-bit = 512-bit wide)
and internal control logic for handshaking.

| Subcategory | Score | % of MT | What it contains |
|-------------|-------|---------|------------------|
| **Compute** (Datapath) | **2,971.23** | 18% | Address increment logic (4 counters for v0..v3 indices); MUX for port arbitration; comparators for loop bounds; index arithmetic for 2D tile mapping |
| **Storage** (Register) | **13,208.50** | 82% | I/O staging registers for 4 × 512-bit external ports; control state registers (handshake token counters); address registers; intermediate pipeline buffers for memory-to-FIFO path |
| **Communication** | **0** | — | Top-level FIFOs; `ccs_in`/`ccs_out` ports are 0-area |
| **MT Total** | **16,179.73** | 100% | |

**Observation**: MT register area (13K) exceeds its datapath (3K) by 4.4×, reflecting the
high register cost of staging 4 wide external memory buses (4 × 512 bits = 2 Kbits of staging).

---

### 4.4 Interconnect — FIFO Area Breakdown

All 8 `ccs_pipe` FIFOs are at the top level (`/top_decoupled_2x1`). Each is 32-bit wide.

| FIFO | Direction | Depth | Area Score | Notes |
|------|-----------|-------|------------|-------|
| `v184:cns` | MT → CT0 | 16 | 4,279.155 | **Data stream** (16-element burst, full buffering) |
| `v185:cns` | MT → CT1 | 16 | 4,279.155 | **Data stream** (same for CT1) |
| `v186:cns` | MT ↔ CT0 | 3 | 862.281 | **Control** (grant token, try_put/try_get) |
| `v187:cns` | MT ↔ CT1 | 3 | 862.281 | **Control** (grant token, try_put/try_get) |
| `v188:cns` | CT0 → MT | 2 | 594.043 | **Result stream** (CT0 output) |
| `v189:cns` | CT1 → MT | 2 | 594.043 | **Result stream** (CT1 output) |
| `v190:cns` | CT ↔ MT | 2 | 594.043 | **Handshake** (done/ack) |
| `v191:cns` | CT ↔ MT | 1 | 325.805 | **Token** (small control) |
| **Total** | | | **12,390.81** | 19% of design total |

#### Communication Area Subcategory

| Type | FIFOs | Total Area | % of Design | Notes |
|------|-------|-----------|-------------|-------|
| **Data** (MT→CT) | 2 × depth-16 | **8,558.31** | 13% | Dominant; 1 per CT, full pipeline depth |
| **Result** (CT→MT) | 2 × depth-2 | **1,188.09** | 1.8% | Small; results sent sequentially |
| **Control/Token** | 2×depth-3 + 1×depth-2 + 1×depth-1 | **2,644.41** | 4.0% | Handshake grant/ack protocol |
| **Total interconnect** | | **12,390.81** | **18.9%** | |

**Key insight**: Data FIFOs (depth-16) account for 69% of total FIFO area because they must
buffer a full 16-element burst. Reducing burst size or using a shared bus would cut FIFO area
significantly. Control FIFOs are cheap (depth 1–3) since they carry single tokens.

---

### 4.5 Whole-Design Area Summary by Subcategory

| Subcategory | Area Score | % | Sources |
|-------------|-----------|---|---------|
| **Compute** | ~15,465 | 24% | CT Datapath (×2) + MT Datapath + FUNC + LOGIC |
| **Storage** | ~34,697 | 53% | CT Register (×2) + MT Register + pipeline buffers |
| **Communication** | ~12,391 | 19% | All 8 ccs_pipe FIFOs (data + control + result) |
| **Other** (lib, FSM) | ~909 | 1% | leading_sign primitive, FSM state, mgc_io_sync |
| **Total** | **~65,462** | 100% | |

---

## 5. Errors Encountered and Fixes

### 5.1 CIN-291: `float` Not Synthesizable

**Error**: `nangate-45nm_beh` does not support native `float`. Catapult rejects FP operations on `float` type.

**Root cause**: The default Allo Catapult emitter used `float` for F32 types.

**Fix** (`mlir/lib/Translation/EmitCatapultHLS.cpp`):
```cpp
// getCatapultTypeName(): F32 maps to ac_ieee_float<binary32>
else if (llvm::isa<Float32Type>(valType))
  return SmallString<16>("ac_ieee_float<binary32>");
```
Added `#include <ac_std_float.h>` to the emitted header.

---

### 5.2 CRD-415: Double Literal → `ac_ieee_float<binary32>` Conversion

**Error**: Array initializers like `{0.000000e+00, ...}` are `double` literals — Catapult EDG front-end cannot implicitly convert `double` to `ac_ieee_float<binary32>`.

**Fix** — Added virtual hook `emitFloatArrayElement` to `VhlsModuleEmitter` base class, overridden in `CatapultModuleEmitter` to append `f` suffix:
```cpp
void CatapultModuleEmitter::emitFloatArrayElement(float value) override {
  if (std::isfinite(value))
    os << std::to_string(value) << "f";   // 0.000000f, not 0.000000e+00
  else if (value > 0) os << "INFINITY";
  else os << "-INFINITY";
}
```

---

### 5.3 CRD-413: `ac_ieee_float<binary32>` → `float` Assignment

**Error**: Stateful global arrays declared as `float[]` but assigned `ac_ieee_float<binary32>` values from the initializer.

**Fix** — Added virtual hook `emitStatefulGlobalElementType` to base class, overridden in `CatapultModuleEmitter`:
```cpp
void CatapultModuleEmitter::emitStatefulGlobalElementType(Type type) override {
  os << getCatapultTypeName(type);   // ac_ieee_float<binary32> for F32
}
```
Applied at both call sites: `emitGlobal()` (module-level) and the stateful-globals section of `emitFunction()`.

---

### 5.4 Scalar Float Literals: `(float)1.000000` → `1.000000f`

**Error**: `Utils.cpp` emitted scalar float constants as `(float)1.000000` (double literal with C-style cast). Catapult interprets the double literal as `double` before the cast, which creates `(ac_ieee_float<binary32>)1.000000` — a double-to-ac_ieee_float conversion rejected by EDG.

**Fix** (`mlir/lib/Translation/Utils.cpp`):
```cpp
if (bitwidth == 32)
  return SmallString<8>(std::to_string((float)value) + "f");   // 1.000000f
```

---

### 5.5 HIER-6: Non-Static Local `ac_channel`

**Error**: Catapult HIER-6 warning/error for local `ac_channel` variables that are not `static`. Catapult requires channels used across function boundaries (or across scheduling regions) to be `static` so they retain state.

**Fix** (`mlir/lib/Translation/EmitCatapultHLS.cpp`):
```cpp
void CatapultModuleEmitter::emitStreamConstruct(allo::StreamConstructOp op) {
  indent();
  os << "static ";   // required by Catapult HIER-6
  Value result = op.getResult();
  ...
}
```

---

### 5.6 HIER-47 / ASSERT-1: FIFO Depth = 0 (Flat Synthesis)

**Error**: Internal Catapult assertion `cap >= 0` in `sif_ap_bif.cxx:1745` triggered when FIFO depth was computed as 0 or negative. This happens because in **flat synthesis** (all sub-functions inlined into top), the scheduler sees channels as intra-process variables and computes their required buffering as 0 or negative (underflow).

**Failed workarounds**:
- Setting `directive set /top_decoupled_2x1/v184:cns -FIFO_DEPTH 2` (accepted but ASSERT-1 still fires)
- Setting depth via `ac_channel<T, depth>` template parameter (rejected: CRD-443, only one template arg)
- `solution options set Message/ErrorOverride HIER-10 -remove` (bypasses warning, not the assertion)

**Root cause**: HIER-47 is a symptom of HIER-10 (cross-hierarchical channel). In flat synthesis, Catapult treats the channel as local → HIER-10 → scheduler underestimates required depth → ASSERT-1.

**Fix**: **Block synthesis** — `solution design set <fn> -block` for each sub-function causes Catapult to keep them as separate RTL blocks. Channels now cross real hierarchical boundaries. Catapult correctly models their buffering requirements and assigns depths ≥ 1. No ASSERT-1.

---

### 5.7 HIER-23: Possible Deadlock (False Positive)

**Warning**: `Possible deadlock detected during symbolic simulation analysis (HIER-23)`

**Status**: False positive. The sequential calling order (MT → CT0 → CT1) is deadlock-free by construction. The decoupled protocol uses try_put/try_get which are non-blocking. Synthesis completes successfully and RTL is generated.

---

### 5.8 CIN-319: `hls_design dataflow` Warning

**Warning**: `hls_design dataflow` attribute not recognized.

**Status**: Benign informational message. The Catapult backend does not use Vitis `#pragma HLS DATAFLOW`; parallelism is specified via block synthesis. No action required.

---

## 6. Synthesis Methodology Notes

### Why Block Synthesis is Required

Catapult has two synthesis modes for sub-functions:
1. **Inline** (default): sub-function body merged into caller's schedule. Local channels treated as variables. HIER-10 triggered if they cross hierarchical source-level boundaries.
2. **Block** (`-block`): sub-function compiled as separate RTL module with I/O interfaces. Channels become real hierarchical FIFO ports. Catapult infers correct FIFO depths via II analysis.

For this design, block synthesis is mandatory because `ac_channel` objects are declared in `top_decoupled_2x1` and passed by reference to `memory_tile_2x1_0`, `compute_tile_2x1_0`, `compute_tile_2x1_1`. These are inherently cross-hierarchical — they must remain as separate RTL blocks.

### Block Synthesis TCL Directives (via `allo/backend/catapult.py`)

The Python backend reads `sub_funcs` from the config dict and emits:
```tcl
go analyze
solution design set memory_tile_2x1_0 -block
solution design set compute_tile_2x1_0 -block
solution design set compute_tile_2x1_1 -block
go compile
```

### `ac_ieee_float<binary32>` vs `float`

The `nangate-45nm_beh` library requires IEEE-754 synthesizable float types from `<ac_std_float.h>`. All three places where `float` could appear in the emitted C++ must be replaced:
1. Variable/return type declarations → `getCatapultTypeName()`
2. Stateful global array element type → `emitStatefulGlobalElementType()` virtual hook
3. Array initializer literals → `emitFloatArrayElement()` virtual hook
4. Scalar float constants → `Utils.cpp` `getFloatValue()` with `f` suffix

### FIFO Depth Inference

With block synthesis, Catapult automatically infers FIFO depths based on the II mismatch between producer and consumer:
- MT throughput = 69 cycles; CT throughput = 298 cycles → CT is 4.3× slower
- Data FIFOs (MT→CT): depth=16 selected (accommodates full burst before CT can consume)
- Control FIFOs: depth=3 (small, handshake tokens)
- Result FIFOs (CT→MT): depth=2 (CT sends 1 result at a time; MT drains quickly)

---

## 7. Files Modified

| File | Change |
|------|--------|
| `mlir/lib/Translation/EmitCatapultHLS.cpp` | F32 → `ac_ieee_float<binary32>`; `static` channels; `emitStatefulGlobalElementType`; `emitFloatArrayElement`; `ac_std_float.h` include |
| `mlir/include/allo/Translation/EmitVivadoHLS.h` | Added virtual hooks `emitStatefulGlobalElementType`, `emitFloatArrayElement` |
| `mlir/lib/Translation/EmitVivadoHLS.cpp` | Default implementations of virtual hooks; both `emitGlobal` call sites updated |
| `mlir/lib/Translation/Utils.cpp` | Scalar F32 constants emit `1.000000f` (with `f` suffix) |
| `allo/backend/catapult.py` | Block synthesis TCL directives from `sub_funcs` config key |
| `catapult_decoupled_2x1.prj/run.tcl` | Block synthesis directives for all 3 tile functions |

---

## 8. Running Synthesis

```bash
# Codegen only (no Catapult needed)
./run_allo.sh python tests/dataflow/catapult_synth_decoupled_2x1.py --mode codegen

# Full synthesis (requires: module load mentor-Catapult_synthesis_10.5a)
./run_allo.sh python tests/dataflow/catapult_synth_decoupled_2x1.py --mode csyn

# PPA extraction from area.rpt
./run_allo.sh python tests/dataflow/catapult_synth_decoupled_2x1.py --mode ppa
```

Project directory: `catapult_decoupled_2x1.prj/`
RTL output: `catapult_decoupled_2x1.prj/Catapult_25/top_decoupled_2x1.v1/rtl.v`
Cycle report: `catapult_decoupled_2x1.prj/Catapult_25/top_decoupled_2x1.v1/cycle.rpt`
Area report: `catapult_decoupled_2x1.prj/Catapult_25/top_decoupled_2x1.v1/rtl.rpt`
