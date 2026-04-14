<!--- Copyright Allo authors. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# ASIC HLS Backend Exploration

## Summary

Explored two commercial ASIC HLS backends (Catapult HLS, Tapa) as part of the
mesh accelerator research. Decision: not pursuing further. Vitis HLS remains the
primary backend. CIRCT is the recommended long-term direction for ASIC targets.

## Catapult HLS (Siemens EDA)

### What was built
- `mlir/lib/Translation/EmitCatapultHLS.cpp` (~561 lines): full C++ emitter, subclassing
  the Vivado emitter and overriding type names, stream API calls, and float handling.
- `allo/backend/catapult.py`: Python-side TCL script generation and host code codegen.
- Key overrides vs Vivado:
  - F32 → `ac_ieee_float<binary32>` (nangate-45nm requires this, not plain `float`)
  - `ac_channel<T>` for streams with `.nb_read()`, `.nb_write()`, `.empty()`
  - `static` prefix on local channel declarations (fixes HIER-6)
  - Block synthesis mode to allow channels to cross hierarchical boundaries

### Synthesis results (Catapult 2024.2, nangate-45nm_beh, 500 MHz / 2ns)
Design: `top_decoupled_2x1` (1 MT + 2 CTs, M=N=K=2, 16 elements)
- CT0/CT1 latency: 295 cycles each; MT: 67 cycles; Total sequential: 657 cycles
- Throughput: 298 cycles (CT-bound)
- Area scores (Catapult internal): CT0=14991, CT1=14991, MT=16180

### Key issues resolved during development
- HIER-10: Local channels → solved by block synthesis
- HIER-47/ASSERT-1: FIFO_DEPTH=0 → solved by block synthesis
- CIN-291: float→fixed-point conversion → use `ac_ieee_float<binary32>`
- CRD-415/CRD-413: double literal / assignment issues → emit `0.000000f`
- HIER-6: Non-static local channel → add `static` prefix

### Why not continuing
- Market niche (automotive/defense, Siemens-adjacent shops)
- Not a standard research community reference tool
- Cadence Stratus is stronger competitor for ASIC research citations
- CIRCT (MLIR-native, Google/Intel-backed) has better long-term trajectory

## Tapa HLS (non-blocking stream additions)

### What was added
- `emitStreamTryGet`, `emitStreamTryPut`, `emitStreamEmpty`, `emitStreamFull` overrides
  in `EmitTapaHLS.cpp`: maps to `.try_read()`, `.try_write()` (Tapa API)
- One test: `test_nb_ops_tapa_codegen` in `tests/dataflow/test_stream_nb_simple.py`

### Why removed
- Tapa is not used in our mesh research flow
- NB stream semantics are fully validated via Vitis HLS (primary target)
- Keeping dead codepath creates maintenance burden in EmitTapaHLS.cpp

## Reference
- Full synthesis report: see git history, commit `01f25e2`
- HLS_SYNTH_REPORT.md in notes/ has Vitis HLS numbers for comparison
