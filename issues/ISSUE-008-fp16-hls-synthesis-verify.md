# ISSUE-008: Verify float16 arithmetic synthesizes in Vitis HLS

**Status**: DONE
**Priority**: High — unblocks ISSUE-009 and ISSUE-010
**Upstream**: No (local verification; results inform PR #578 description update)

---

## Problem

PR #578 adds float16 host-side type mappings (`"f16" → "half"`) and claims to fix
float16 HLS support, but no synthesis test exists. It is unverified that a float16
kernel actually synthesizes through Vitis HLS 2023.2 without errors.

---

## Plan

Create `tests/dataflow/hls_synth_fp16.py` (csyn-only, no csim) with two kernels:

**Test A — basic arithmetic** (`top_fp16_arith`):
- Simple float16 array kernel: elementwise add + scale
- Verifies `half` type compiles and synthesizes at all
- Project: `/scratch/sk3463/hls_projects/fp16_arith_csyn.prj`

**Test B — exp** (`top_fp16_exp`):
- Float16 array kernel calling `allo.exp()` on each element
- Verifies whether `exp(half)` synthesizes via `hls_math.h`
- Project: `/scratch/sk3463/hls_projects/fp16_exp_csyn.prj`

Follow the exact pattern of `tests/dataflow/hls_synth_streams.py`:
- Use `df.build(..., target="vitis_hls", mode="csyn", project=...)`
- Check `hls.is_available("vitis_hls")` and skip gracefully if not
- Output project dirs to `/scratch/sk3463/hls_projects/` (not repo, avoids quota)

---

## Acceptance Criteria

- [x] Script runs without Python errors
- [x] Test A csyn completes without HLS errors (confirms `half` is synthesizable)
- [x] Test B csyn outcome documented (pass → note; fail → triggers ISSUE-010)
- [ ] PR #578 description updated with actual verified status (pending upstream PR)

## Results (2026-04-13, Vitis HLS 2023.2, U280, 411 MHz)

**Test A — `top_fp16_arith` (add + scale): PASS**

| LUT | FF | BRAM_18K | DSP | Latency (cycles) | II | Fmax |
|-----|----|----------|-----|------------------|----|------|
| 3986 | 4233 | 6 | 5 | 46–48 | 21 | 411 MHz |

HLS generates `hptosp`/`sptohp` half↔float conversion cores for arithmetic.
The `half` type synthesizes correctly — PR #578's `"f16" → "half"` mapping is verified.

**Fixes applied during verification:**
1. `allo/ir/builder.py`: Added `F16Type`, `BF16Type` to imports and the scalar
   `math_d.*` dispatch type guard (line ~3261). Without this, `allo.exp(float16_val)`
   raised `AttributeError: 'Float' object has no attribute '__name__'` at the
   Python→MLIR lowering stage.

**Test B — `top_fp16_exp` (allo.exp per element): PASS** (after ISSUE-010 fix)

| LUT | FF | BRAM_18K | DSP | Latency (cycles) | II |
|-----|----|----------|-----|------------------|----|
| 2668 | 2576 | 4 | 2 | 38–40 | 13 |

Initial run failed with `[HLS 207-3320] call to 'exp' is ambiguous`. Fixed in
ISSUE-010 by emitting `hls::exp(x)` instead of `exp(x)` for `Float16Type` operands
in `EmitVivadoHLS.cpp`. `hls_math.h` provides `exp(half)` in `namespace hls`;
qualification resolves the ambiguity.

---

## Dependencies

- Requires: `allo` conda env with `vitis_hls` available (brg-zhang-xcel only)
- Blocks: ISSUE-009 (safe to expose scalar exp), ISSUE-010 (conditional emitter fix)
