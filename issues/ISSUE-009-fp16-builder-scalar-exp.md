<!--- Copyright Allo authors. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# ISSUE-009: Fix scalar exp dispatch in builder.py for float16

**Status**: DONE
**Priority**: Medium — blocked on ISSUE-008 confirming synthesis safety
**Upstream**: Yes — PR against cornell-zhang/allo after verification

---

## Problem

`allo/ir/builder.py:3261` restricts the scalar math function dispatch (exp, log, sqrt, etc.)
to `(F32Type, F64Type, IntegerType)`. Calling `allo.exp(x)` where `x: float16` raises
`RuntimeError: Unsupported function exp with type [F16Type(f16)]` at Python build time,
before any HLS codegen is even attempted.

The tensor/array path (`linalg_d.exp`) has no such restriction and should already work
for float16 arrays, but the scalar path is completely blocked.

---

## Fix

**File**: `allo/ir/builder.py`

1. Add `F16Type` to the import block (where `F32Type`, `F64Type` are imported).
2. Add `F16Type` to the type check at line ~3261:

```python
# Before
if all(
    isinstance(arg_type, (F32Type, F64Type, IntegerType))
    for arg_type in arg_types
):
# After
if all(
    isinstance(arg_type, (F32Type, F64Type, F16Type, IntegerType))
    for arg_type in arg_types
):
```

Same change applies to all scalar math ops dispatched from the same block
(log, sqrt, sin, cos, tanh, etc.) — they share one type check.

Before submitting upstream, verify the pattern matches how existing float type
support is structured in the file (check imports, check if BF16Type is handled
elsewhere consistently).

---

## Acceptance Criteria

- [x] `allo.exp(x: float16)` no longer raises at Python build time
- [ ] Existing float32/float64 exp tests still pass (run on zhang-21 before upstream PR)
- [ ] Linting clean (`bash scripts/lint/task_lint.sh`)
- [ ] Upstream PR opened (pending)

## Applied Fix (2026-04-13)

`allo/ir/builder.py` — also added `BF16Type` for consistency:
```python
# imports: added F16Type, BF16Type alongside F32Type, F64Type
# type guard (line ~3263):
isinstance(arg_type, (F16Type, BF16Type, F32Type, F64Type, IntegerType))
```
`allo.exp(float16_val)` now lowers to `math.ExpOp` with `f16` operand correctly.
The synthesis failure (ISSUE-010) is in the HLS C++ emitter, not here.

---

## Dependencies

- Blocked on: ISSUE-008 (confirm synthesis doesn't reveal a deeper issue)
- Does not block: ISSUE-010 (independent emitter fix)
