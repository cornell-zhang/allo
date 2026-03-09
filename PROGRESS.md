# Allo Mesh Accelerator — Progress Log

## Summary Status

| Component | Status | Notes |
|-----------|--------|-------|
| Non-blocking stream primitives (IR) | ✅ DONE | try_put, try_get, empty, full |
| Simulator backend | ✅ DONE | Lock-free FIFO lowering |
| Vitis HLS backend | ✅ DONE | .read_nb / .write_nb / .empty / .full |
| Tapa HLS backend | ✅ DONE | .try_read / .try_write |
| Layer tests (IR, sim, HLS codegen) | ✅ DONE | 12/12 tests pass |
| Decoupled mesh demo (1 MT + 1 CT) | ✅ DONE | test_decoupled_mesh.py |
| Hierarchical decoupled 2×1 mesh | ✅ DONE | test_decoupled_mesh.py::test_decoupled_2x1_mesh |
| Blocking demo (reference) | ✅ DONE | test_hierachical_mesh.py (2x1, 2x2) |
| HLS synthesis (blocking vs NB) | ✅ DONE | hls_synth_streams.py; HLS_SYNTH_REPORT.md |
| HLS synthesis (decoupled mesh) | ✅ DONE | hls_synth_decoupled.py; HLS_SYNTH_REPORT.md |
| RTL/MatchLib backend | ⬜ TODO | Future work |
| 2×2 decoupled mesh | ⬜ TODO | Extend test_decoupled_mesh.py |

---

## Session Log

### Session: Non-Blocking Stream Semantics + Decoupled Mesh Demo

**Context**: Building on prior work (hierarchical mesh with blocking FIFOs, peer topology
fixes), this session extended Allo with non-blocking stream primitives and validated them
at every abstraction level.

**Work Completed**:

1. **Language Extension** (`allo/ir/types.py`, `allo/ir/infer.py`, `allo/ir/builder.py`):
   - Added `try_put(val) -> int1`, `try_get() -> (T, int1)`, `empty() -> int1`, `full() -> int1`
     as methods on the `Stream` type's Python wrapper.
   - Extended the MLIR type inferencer to handle the 2-result tuple from `try_get`.
   - Extended the AST builder to emit the corresponding MLIR ops.

2. **MLIR Dialect** (`mlir/include/allo/Dialect/AlloOps.td`):
   - Defined `StreamTryPutOp`, `StreamTryGetOp`, `StreamEmptyOp`, `StreamFullOp`.
   - Added visitor dispatch (`mlir/include/allo/Dialect/Visitor.h`).

3. **Simulator Backend** (`allo/backend/simulator.py`):
   - Non-blocking ops lowered to lock-free FIFO access using `scf.if` + atomic head/tail pointer updates.
   - Empty/Full lowered to inline comparisons (no spin loop, O(1) latency).
   - **Bug Fixed (#6)**: `StreamPutOp` in callee-arg section used wrong `ip=before_ip`
     after `ConditionOp` terminator; changed to `ip=replace_ip`.

4. **HLS Backends** (Vitis + Tapa):
   - Vitis: `write_nb`, `read_nb`, `.empty()`, `.full()` API calls.
   - Tapa: `try_write`, `try_read` API calls.

5. **Test Suite** (all pass, 11/11):
   - `test_stream_ops_ir.py`: MLIR op emission checks
   - `test_stream_ops_sim.py`: Simulator functional correctness
   - `test_stream_ops_hls.py`: HLS code-generation pattern checks
   - `test_stream_nb_simple.py`: Comprehensive end-to-end tests + HLS cost docs
   - `test_decoupled_mesh.py`: Decoupled valid-ready handshake + burst streaming demo

6. **Documentation**:
   - `ALLO_CHANGE.md`: Updated with changes #5 (non-blocking primitives) and #6 (IP bug fix)
   - `PLAN.md`: Architecture plan and future directions
   - `PROGRESS.md`: This file

**Bugs Fixed This Session**:
- `allo/backend/simulator.py` line ~635: `ip=before_ip` → `ip=replace_ip` for
  blocking `StreamPutOp` data-write ops after spin-while condition terminator
  (Change #6 in ALLO_CHANGE.md).

---

## Pending / Next Steps

### Short Term
- [ ] Run Vitis HLS synthesis (mode="hw") to get actual LUT/FF/II numbers
  for blocking vs non-blocking stream variants. Document in `test_stream_nb_simple.py`.
- [ ] Extend `test_decoupled_mesh.py` to a full 2×1 mesh (2 CTs + 1 MT)
  matching the architecture of `test_hierachical_mesh.py`.
- [ ] Test `test_hierachical_mesh.py` `test_2x1()` end-to-end on simulator.

### Medium Term
- [ ] Implement credit-based flow control as an alternative to valid-ready polling
  (integer credit counter streams instead of try_put/try_get busy-wait).
- [ ] Design scalable routing tables for parameterized mesh dimensions.

### Long Term
- [ ] Explore MatchLib (Catapult) integration for cycle-accurate timing modeling.
- [ ] RTL backend: emit Verilog/SystemVerilog directly for timing-critical interconnect
  primitives that HLS tools cannot schedule optimally.
- [ ] Formal verification of deadlock-freedom for non-blocking stream protocols.

---

## Architecture Notes for Next Agent

### Non-Blocking Semantics Summary
```python
# Valid-Ready handshake protocol:
# Producer side (MT) — raise valid:
sent = ctrl_stream.try_put(MSG_TYPE)   # returns True if CT accepted

# Consumer side (CT) — poll for valid:
msg, has_req = ctrl_stream.try_get()   # returns (val, True) if data present

# Backpressure checks:
e = ctrl_stream.empty()  # True if no data
f = data_stream.full()   # True if no space

# Pattern: handshake then burst
while not ctrl_stream.try_put(msg_type):
    pass  # spin until CT accepts
for i in range(BURST_SIZE):
    data_stream.put(data[i])  # blocking burst after handshake
```

### Key Files
- `allo/ir/types.py`: Stream type Python API
- `allo/ir/builder.py`: AST→MLIR, look for `build_Call` / stream method handling
- `allo/backend/simulator.py`: `_process_function_streams()` — main lowering logic
- `mlir/include/allo/Dialect/AlloOps.td`: Dialect op definitions
- `mlir/lib/Translation/EmitVivadoHLS.cpp`: HLS emission
- `tests/dataflow/test_decoupled_mesh.py`: End-to-end decoupled demo

### Known Constraints
- Scalars in `df.region` must be `int32[1]` not `int32` (MLIR memref lowering)
- Dynamic stream array indexing fails; manual unrolling required
- All `@df.kernel` names must be globally unique across the module
- HLS CSIM/COSIM: run manually with `df.build(top, target="vitis_hls", mode="csim")`
- Non-blocking try_get returns a tuple; assign with `val, ok = stream.try_get()`

### Future Backend Targets
The current implementation supports `target="simulator"`, `target="vitis_hls"`,
and `target="tapa"`. For timing-critical or cycle-accurate interconnect:
- **MatchLib / Catapult**: cycle-accurate FIFO, arbiter, credit controller models
- **RTL backend**: direct Verilog/SystemVerilog emission for custom interconnect
- Allo's translation infrastructure is in `mlir/lib/Translation/`; add new EmitXxx.cpp
