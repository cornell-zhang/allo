# ISSUE-010: Conditional emitter fix for exp(half) in Vitis HLS

**Status**: OPEN (conditional — only proceed if ISSUE-008 Test B fails)
**Priority**: Low — only activated if synthesis reveals a problem
**Upstream**: Yes — part of or follow-up to PR #578

---

## Problem

The Vitis HLS emitter (`EmitVivadoHLS.cpp`) uses a generic `emitUnary(op, "exp")` for all
float types, blindly emitting `result = exp(operand);`. For `half` operands, this relies on
`hls_math.h` providing an ADL-visible overload `exp(half)`.

AMD's `hls_math.h` provides `hls::half_exp()` as the canonical form. Whether the
unqualified `exp(half)` call works depends on whether that overload is pulled into the
global namespace. If ISSUE-008 Test B (exp csyn) fails with a type resolution error,
the emitter needs to special-case `Float16Type + math::ExpOp`.

**Do not implement this unless ISSUE-008 confirms the fix is needed.**

---

## Fix (if needed)

**File**: `mlir/lib/Translation/EmitVivadoHLS.cpp`

Override the `math::ExpOp` visitor to emit `hls::half_exp` when the operand is `half`:

```cpp
void visitOp(math::ExpOp op) override {
  if (op.getOperand().getType().isa<Float16Type>())
    emitter.emitUnary(op, "hls::half_exp");
  else
    emitter.emitUnary(op, "exp");
}
```

Before implementing, check:
- How other type-specific overrides are done in `EmitVivadoHLS.cpp` (follow the same pattern)
- Whether similar overrides exist for `hls::half_log`, `hls::half_sqrt`, etc. (apply consistently)
- AMD UG902 / `hls_math.h` docs for the full list of `half`-specific math function names

---

## Acceptance Criteria

- [ ] ISSUE-008 Test B confirmed failing (prerequisite to even start)
- [ ] `exp(half)` synthesizes without errors after fix
- [ ] No regression on float32/float64 exp synthesis
- [ ] Consistent treatment for other half math ops (log, sqrt, etc.) if same issue exists

---

## Dependencies

- Blocked on: ISSUE-008 Test B result
- Independent of: ISSUE-009
