# float16 Support in Vitis HLS Backend

## Summary

float16 (`half`) is a first-class type in Vitis HLS (UG902, §Half-Precision Floating-Point).
The kernel-side C++ emitter (`EmitVivadoHLS.cpp`) already maps `Float16Type → "half"`.
The host-side type mappings (`ctype_map`, `allo2c_type`) were added in commit `ce1e0b3`
and sent upstream as PR #578.

bfloat16 kernel-side emitter support is **not implemented** (no `BF16Type` case in
`EmitVivadoHLS.cpp`). bfloat16 is fully supported only in the AIE backend. Deferred.

---

## Status (as of 2026-04-13)

| Layer | Status | Location |
|-------|--------|----------|
| Type definition | ✅ `Float(16, 10, "f16")` | `allo/ir/types.py:431` |
| Kernel C++ type name | ✅ emits `half` | `mlir/lib/Translation/EmitVivadoHLS.cpp:35` |
| Host `ctype_map` | ✅ `"f16" → "half"` | `allo/backend/vitis.py:56` |
| Host `allo2c_type` | ✅ `"float16" → "half"` | `allo/utils.py:78` |
| Simulator | ✅ bit-cast via `c_int16` | `allo/backend/llvm.py:157` |
| Arithmetic in HLS | ❓ emits correctly; synthesis **unverified** | ISSUE-008 |
| Scalar `exp` in builder | ❌ blocked by F32/F64-only type check | `allo/ir/builder.py:3261`, ISSUE-009 |
| `exp(half)` in HLS synthesis | ❓ emits `exp(x)`; depends on `hls_math.h` | ISSUE-010 |

---

## How `half` is Emitted

### Kernel side

```cpp
// mlir/lib/Translation/EmitVivadoHLS.cpp:35
static SmallString<16> getTypeName(Type valType) {
  if (llvm::isa<Float16Type>(valType))
    return SmallString<16>("half");  // UG902 p.222
  ...
}
```

Standard headers always included in emitted kernels:
```cpp
#include <ap_fixed.h>   // defines `half`
#include <hls_math.h>   // provides exp(half), etc.
#include <hls_stream.h>
```

No extra header needed — `ap_fixed.h` already brings in `half`.

### Unary math ops

The emitter uses a generic `emitUnary(op, "exp")` for all float types, emitting
`result = exp(operand);` regardless of type. For `half` operands, correctness depends
on `hls_math.h` providing an ADL-visible `exp(half)` overload (which AMD provides,
but **synthesis has not been run to confirm**).

---

## Known Gaps

### 1. Scalar `exp` blocked in `builder.py`

`allo/ir/builder.py:3261` restricts the scalar math dispatch to `(F32Type, F64Type, IntegerType)`.
Calling `allo.exp(x)` where `x: float16` raises `RuntimeError` at Python build time,
before any HLS codegen is attempted.

Fix: add `F16Type` to the tuple (and import). See ISSUE-009.

### 2. Synthesis verification missing

No HLS synthesis test exists for float16. PR #578 adds host-side type mappings but does
not verify that a float16 kernel synthesizes through Vitis HLS 2023.2 end-to-end.

See ISSUE-008.

### 3. `exp(half)` synthesis may require qualified form

If `hls_math.h` requires `hls::half_exp()` rather than unqualified `exp()`, the emitter
needs to special-case `Float16Type + math::ExpOp`. This is a conditional fix depending
on ISSUE-008 results.

See ISSUE-010.

---

## References

- AMD UG902 Vitis HLS User Guide, §Half-Precision Floating-Point, p. 222
- `hls_math.h` half API: `hls::half_exp`, `hls::half_log`, `hls::half_sqrt`, etc.
- Upstream issue: cornell-zhang/allo#478
- Upstream PR: cornell-zhang/allo#578
