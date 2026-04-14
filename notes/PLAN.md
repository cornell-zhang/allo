# Allo Mesh Accelerator — Project Plan

## Vision
Build a template programmable dataflow domain-specific accelerator (DSA) architecture
within the Allo HLS framework, exploring flexible and performant on-chip interconnect
semantics beyond the blocking FIFO model.

## Background: Why Extend Allo's Stream Semantics?

Allo's `Stream[T, depth]` type maps to blocking FIFOs (like `hls::stream`). This means
every `.put()` stalls until space is available and every `.get()` stalls until data arrives.
For multi-tile architectures, this forces strict **fixed-II protocols**: both sides must
always execute exactly the same number of puts/gets per invocation — even sending garbage
NOPs to maintain balance. This is shown in `tests/dataflow/test_hierachical_mesh.py`.

The limitation: **truly independent control flows** in each tile, communicating only on
demand via conditional handshake, cannot be expressed with blocking-only streams.

## Proposed: Handshaked Non-Blocking Streams

Standard on-chip interconnect literature (AXI-S, NoC, credit-based flow control, Hwacha,
NVIDIA NVLink) uses **valid-ready** handshake:
- Producer raises `valid` (try_put); Consumer raises `ready` (try_get)
- Handshake completes when both are high simultaneously
- Data burst streaming proceeds unconditionally after handshake

We implement this as four new primitives:
```python
ok      = stream.try_put(val)    # non-blocking write (valid signal)
val, ok = stream.try_get()       # non-blocking read  (ready polling)
empty   = stream.empty()         # predicate: no data in FIFO
full    = stream.full()          # predicate: FIFO at capacity
```

## Architecture Target

**2×1 and 2×2 mesh** of compute tiles (CTs) + memory tiles (MTs):
- MT dispatches commands via control streams (try_put/try_get for handshake)
- CT responds with burst data transfers via unconditional blocking streams
- CTs can run independently, calling sub-regions (gemm, vadd) via nested df.region calls
- Reference blocking demo: `tests/dataflow/test_hierachical_mesh.py`
- Decoupled demo: `tests/dataflow/test_decoupled_mesh.py`

## Implementation Layers

### Layer 1: Type System & IR (DONE)
- `allo/ir/types.py`: Python methods `.try_put()`, `.try_get()`, `.empty()`, `.full()`
- `allo/ir/infer.py`: TypeInferer for multi-result `try_get` ops
- `allo/ir/builder.py`: AST→MLIR translation for all four ops
- `mlir/include/allo/Dialect/AlloOps.td`: MLIR op definitions
- `mlir/include/allo/Dialect/Visitor.h`: Visitor dispatch

### Layer 2: Simulator Backend (DONE)
- `allo/backend/simulator.py`: Lock-free FIFO lowering using atomic head/tail updates
- Non-blocking paths use `scf.if` instead of `scf.while` spin loops

### Layer 3: Vitis HLS Backend (DONE)
- `mlir/lib/Translation/EmitVivadoHLS.cpp`: Maps to `.read_nb()`, `.write_nb()`, `.empty()`, `.full()`

### Layer 4: Tapa HLS Backend (DONE)
- `mlir/lib/Translation/EmitTapaHLS.cpp`: Maps to `.try_read()`, `.try_write()`

### Layer 5: Future — MatchLib / RTL Backend (TODO)
- For timing-critical interconnect semantics that HLS cannot model well
- MatchLib (Catapult) provides cycle-accurate FIFO and credit-counter models
- A new `RTL` backend target could emit pure Verilog/SystemVerilog
- Useful for: multi-cycle routing (crossbars, mesh NoC), credit-based flow control,
  wormhole routing, virtual channels

## Scalability Architecture Design

```
                    ┌──────────────────────────────────────────┐
                    │              host interface              │
                    └───────────────┬──────────────────────────┘
                                    │
                    ┌───────────────▼──────────────────────────┐
                    │           Memory Tile (MT_0)              │
                    │  try_put/try_get  ──▶  cmd streams        │
                    │  burst put/get    ──▶  data streams       │
                    └──────┬─────────────────────┬─────────────┘
                           │                     │
               ┌───────────▼───┐         ┌───────▼────────────┐
               │  Compute Tile │         │   Compute Tile      │
               │     CT_00     │         │      CT_01          │
               │  (gemm/vadd   │         │  (gemm/vadd         │
               │   sub-regions)│         │   sub-regions)      │
               └───────────────┘         └────────────────────┘
```

For a 2×2 mesh, MT_0 ←try_put/get→ MT_1 for inter-MT coordination.

## Performance / Cost Evaluation (HLS Focus)

### Methodology
1. Generate HLS code for each stream op variant
2. Run Vitis HLS synthesis (not automated; use `mode="hw"`)
3. Measure: LUT count, FF count, II (initiation interval), latency
4. Compare: blocking vs non-blocking for same logical function

### Key Questions
- What is the overhead of `try_put`/`try_get` vs `put`/`get` in LUTs/FFs?
- Does the success-bit MUX on the control path become critical path?
- At what FIFO depth does full/empty comparator area become significant?
- What burst depth is optimal for amortizing the try_put/try_get handshake overhead?

### Expected Results (Hypothesis)
- Non-blocking ops: +2–4 LUTs per stream (one comparator, one MUX for success flag)
- No II penalty (non-blocking ops complete in II=1 unconditionally)
- Handshake overhead amortized when burst size ≥ 8 elements

## Future Directions

1. **Scalable Routing Logic**: Add parameterized routing tables to the MT for multi-hop mesh
2. **Credit-Based Flow Control**: Replace ready polling with integer credit counters
3. **RTL Backend**: New Allo target for expressing timing-critical interconnect patterns (Maybe Filament? It has verification guarantees)
4. **MatchLib Integration**: Use Catapult's cycle-accurate FIFO/arbiter library
5. **Wormhole Routing**: Extend stream type to carry routing headers
