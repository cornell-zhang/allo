<!--- Copyright Allo authors. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# ISSUE-010: Conditional emitter fix for exp(half) in Vitis HLS

**Status**: DONE
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

- [x] ISSUE-008 Test B confirmed failing (prerequisite to even start)
- [x] `exp(half)` synthesizes without errors after fix
- [x] No regression on float32/float64 exp synthesis
- [x] Consistent treatment for other half math ops (log, sqrt, etc.) applied

## Applied Fix (2026-04-13)

**Root cause**: `hls_math.h` declares `exp(half)` inside `namespace hls`, so ADL
should find it, but `<math.h>` is also included bringing in C `exp(double)`, causing
ambiguity. The emitter's `emitUnary(op, "exp")` emits a bare `exp(...)` call.

**Fix in `mlir/lib/Translation/EmitVivadoHLS.cpp`**: Added a static helper `isHalf()`
checking `llvm::isa<Float16Type>` on the operand, and changed all scalar math unary
visitors to emit `hls::exp`, `hls::log`, `hls::sqrt`, `hls::sin`, `hls::cos`,
`hls::tanh`, `hls::exp2`, `hls::log2`, `hls::log10`, `hls::abs` when the operand
is float16 (otherwise fall through to the unqualified name as before).

**Synthesis results** (Vitis HLS 2023.2, U280, 411 MHz):

| Design | LUT | FF | BRAM_18K | DSP | Latency | II |
|--------|-----|----|----------|-----|---------|----|
| `top_fp16_arith` | 3986 | 4233 | 6 | 5 | 46–48 | 21 |
| `top_fp16_exp` | 2668 | 2576 | 4 | 2 | 38–40 | 13 |

**Side fix in `mlir/lib/Translation/EmitTapaHLS.cpp`**: Added missing implementations
and `StmtVisitor` dispatch entries for `emitStreamTryGet`, `emitStreamTryPut`,
`emitStreamEmpty`, `emitStreamFull` — these were declared in the header but never
implemented. Tapa uses `.try_read()`, `.try_write()`, `.empty()`, `false` (no `.full()`).

## Dependencies

- Blocked on: ISSUE-008 Test B result ✓
- Independent of: ISSUE-009
