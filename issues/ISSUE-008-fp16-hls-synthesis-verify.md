# ISSUE-008: Verify float16 arithmetic synthesizes in Vitis HLS

**Status**: OPEN
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

- [ ] Script runs without Python errors
- [ ] Test A csyn completes without HLS errors (confirms `half` is synthesizable)
- [ ] Test B csyn outcome documented (pass → note; fail → triggers ISSUE-010)
- [ ] PR #578 description updated with actual verified status

---

## Dependencies

- Requires: `allo` conda env with `vitis_hls` available (brg-zhang-xcel only)
- Blocks: ISSUE-009 (safe to expose scalar exp), ISSUE-010 (conditional emitter fix)
