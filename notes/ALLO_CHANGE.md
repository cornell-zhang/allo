# Allo Compiler Changes

The following changes were applied to the `allo` compiler codebase to fix bugs discovered during the implementation of the `test_mesh_accelerator.py` dataflow region.

---

### 1. Missing `global_op_cache` on `ASTContext.copy()`
**Definition**: When `df.region()` compiles nested kernels that use `@stateful` streams, it relies on mapping global names to MLIR operations. During compilation, the context for nested blocks is duplicated via `ASTContext.copy()`. However, `copy()` previously failed to selectively copy the `global_op_cache` dictionary, leading to `AttributeError` when the compiler attempted to locate `global_op_cache` for stateful variable generation.

**Reproducing Steps**: 
1. Create a `df.region()`
2. Define a nested kernel inside using `@df.kernel(...)`
3. Declare an internal stream inside the nested kernel using `val: type @ stateful = 0`
4. Compile using `df.build(region_name)`

**How it is fixed**: Explicitly copy the `global_op_cache` attribute if it exists when duplicating the context.

**Files and Lines affected**:
- `allo/ir/visitor.py`
  - In `ASTContext.copy()`: Checked for `hasattr(self, "global_op_cache")` and copied it to the new `ctx` if present.


---

### 2. Scalar Buffer Unwrapping in Nested Region `func.call`
**Definition**: When calling a generic function or nested kernel from inside a `df.region()`, all arguments are internally tracked as `MemRefType` buffers. If the inner kernel accepts a `0-D MemRef` (representing a scalar, like `i32`), MLIR expects a raw `i32` value inside the `func.call` operands. The compiler was erroneously passing the raw pointer `memref<i32>` directly into `func.call`, resulting in an MLIR parsing type mismatch.

**Reproducing Steps**:
1. Define a `df.region()` top-level function that accepts a scalar `ctrl: int32`
2. Define an inner `df.kernel(..., args=[ctrl])` wrapper that tries to pass `ctrl` iteratively to standard MLIR functions via `compute_tile(ctrl)`
3. MLIR fails validation: `'func.call' op operand type mismatch: expected operand type 'i32', but provided 'memref<i32>'`

**How it is fixed**: Intercept the arguments sent to `func.call`. If the returned MLIR operand type is a `MemRefType` and `len(shape) == 0`, invoke `ASTTransformer.build_scalar()` to emit an `AffineLoadOp` to extract the underlying scalar value before executing the function call.

**Files and Lines affected**:
- `allo/ir/builder.py`
  - In `ASTTransformer.build_FunctionDef()`, modified two sites generating `func_d.CallOp`:
    1. During nested kernel argument preparation (around `args_kw`) 
    2. During standard function definition call generation (towards end of function).
    Added shape checking logic to call `ASTTransformer.build_scalar(ctx, arg)` instead of passing raw memrefs.

---

---

### 3. Nested Control-Flow Sub-Region Calls Not Processed by Simulator Backend
**Definition**: `_process_function_streams` in `allo/backend/simulator.py` only scanned the **top-level** operations of a function body for `func.call` ops. When a `df.kernel` calls a sub-`df.region` from inside control flow (e.g., `scf.if` / `scf.for`), those `func.call` ops are nested several levels deep and were never found. As a result, the callee region's `allo.stream_construct` / `allo.stream_get` / `allo.stream_put` ops were never lowered to the FIFO-struct implementation, causing LLVM lowering to fail with `missing LLVMTranslationDialectInterface registration for dialect for op: func.func`.

**Reproducing Steps**:
1. Define two `df.region`s (`gemm_region`, `vadd_region`) as standalone module-level regions.
2. Inside a `df.kernel` (e.g., `idecode`) that is itself nested in a third `df.region` (`compute_tile`), call `gemm_region(...)` or `vadd_region(...)` conditionally inside a `for`/`if` block.
3. `df.build(compute_tile, target="simulator")` raises `RuntimeError: Failure while creating the ExecutionEngine` with the LLVM dialect error above.

**How it is fixed**: After the existing top-level `func.call` scan in `_process_function_streams`, added a recursive scan using `recursive_collect_ops` to discover all `func.call` ops anywhere in the function body (including inside `scf.if`/`scf.for`). For each newly found callee that hasn't been processed yet, `_process_function_streams` is called recursively. The existing `processed_funcs` guard makes repeated visits a cheap no-op. Nested call sites are not added to `pe_call_define_ops` (so no spurious OMP sections are injected at the call site in the parent kernel).

**Files and Lines affected**:
- `allo/backend/simulator.py`
  - In `_process_function_streams()`: Added a block after the top-level `for op in func_ops` loop that calls `recursive_collect_ops(func.operation, (func_d.CallOp,), ...)` and recursively processes any newly discovered callees.

---

### 4. Nested OMP Parallelism Deadlock for Peer Kernels with Sub-Region Calls
**Definition**: When peer `df.kernel`s (siblings inside a single `df.region`) call sub-`df.region`s at runtime, the sub-regions have their own `omp.parallel`/`omp.sections` blocks injected by the simulator backend. However, the peer kernels are already running inside OMP sections from the parent region. By default, OpenMP serializes nested parallelism (`OMP_MAX_ACTIVE_LEVELS=1`), so the sub-region's PE kernels cannot run concurrently, causing a stream deadlock.

**Reproducing Steps**:
1. Define a `df.region` (`top_2x1_peer`) with three peer `df.kernel`s: `memory_tile_peer`, `compute_tile_0`, `compute_tile_1`.
2. `compute_tile_0` calls `gemm_region(...)` or `vadd_region(...)` (sub-`df.region`s) during `CTRL_RUN`.
3. `df.build(top_2x1_peer, target="simulator")` succeeds, but `CTRL_RUN` deadlocks at runtime.

**How it is fixed**: `build_dataflow_simulator` now auto-sets `os.environ["OMP_MAX_ACTIVE_LEVELS"] = "4"` before lowering, unless the user has already set it. This allows nested OMP parallel regions to spawn threads correctly.

**Files and Lines affected**:
- `allo/backend/simulator.py`
  - In `build_dataflow_simulator()`: Added `OMP_MAX_ACTIVE_LEVELS` environment variable setup at the top of the function.

---

### 5. Non-Blocking Stream Primitives (try_put, try_get, empty, full)
**Definition**: Added four new non-blocking stream operations as a language and IR extension, enabling handshaked, decoupled communication between kernels without requiring the strict fixed-II (balanced send/receive count) protocol that blocking `put`/`get` impose. This unlocks valid-ready handshake patterns and credit-based flow control architectures.

**Semantics**:
- `s.try_put(val) -> int1`: Non-blocking write. Returns `True` if the stream had space and the write succeeded; `False` if full. Does not stall.
- `(val, ok) = s.try_get() -> (T, int1)`: Non-blocking read. Returns `(value, True)` if the stream had data; `(zero, False)` if empty. Does not stall.
- `s.empty() -> int1`: Predicate; returns `True` if the stream contains no elements.
- `s.full() -> int1`: Predicate; returns `True` if the stream is at capacity.

**Implementation Layers**:
1. **IR / Type system** (`allo/ir/types.py`, `allo/ir/infer.py`, `allo/ir/builder.py`):
   - Extended `Stream` type's `.try_put()`, `.try_get()`, `.empty()`, `.full()` Python methods to emit `allo.stream_try_put`, `allo.stream_try_get`, `allo.stream_empty`, `allo.stream_full` MLIR ops.
   - Extended `TypeInferer` to handle the multi-result types of `try_get` (returns `(T, i1)`).
2. **MLIR Dialect** (`mlir/include/allo/Dialect/AlloOps.td`, `mlir/include/allo/Dialect/Visitor.h`):
   - Defined `StreamTryPutOp`, `StreamTryGetOp`, `StreamEmptyOp`, `StreamFullOp` ops in the Allo dialect.
   - Added visitor pattern dispatch for all four ops.
3. **Simulator Backend** (`allo/backend/simulator.py`):
   - `StreamEmptyOp` → inline `head == tail` comparison (no spin).
   - `StreamFullOp` → inline `tail_next == head` comparison (no spin).
   - `StreamTryPutOp` → `scf.if(not_full) { write; update_tail; yield true } else { yield false }`.
   - `StreamTryGetOp` → `scf.if(not_empty) { read; update_head; yield (val, true) } else { yield (zero, false) }`.
4. **Vitis HLS Backend** (`mlir/lib/Translation/EmitVivadoHLS.cpp`, `mlir/include/allo/Translation/EmitVivadoHLS.h`):
   - `StreamTryPutOp` → `stream.write_nb(val)` (HLS non-blocking write API).
   - `StreamTryGetOp` → `stream.read_nb(val)` (HLS non-blocking read API).
   - `StreamEmptyOp` → `stream.empty()`.
   - `StreamFullOp` → `stream.full()`.
5. **Tapa HLS Backend** (`mlir/lib/Translation/EmitTapaHLS.cpp`, `mlir/include/allo/Translation/EmitTapaHLS.h`):
   - `StreamTryPutOp` → `stream.try_write(val)`.
   - `StreamTryGetOp` → `stream.try_read(val)`.

**Tests Added**:
- `tests/dataflow/test_stream_ops_ir.py`: IR-level checks (MLIR ops emitted correctly).
- `tests/dataflow/test_stream_ops_sim.py`: Simulator functional correctness.
- `tests/dataflow/test_stream_ops_hls.py`: HLS code-generation pattern checks (Vivado + Tapa).
- `tests/dataflow/test_stream_nb_simple.py`: Comprehensive end-to-end tests across IR, simulator, and HLS codegen levels; includes HLS cost documentation.
- `tests/dataflow/test_decoupled_mesh.py`: Demonstration of valid-ready handshake + burst data streaming replacing fixed-II NOPs.

---

### 6. Blocking StreamPutOp Lowering: Wrong InsertionPoint After Terminator
**Definition**: In `_process_function_streams`, the section handling blocking `StreamPutOp` ops in callee (kernel) functions incorrectly used `ip=before_ip` (inside the `scf.while` `before` block, after its `scf.condition` terminator was already inserted) for the data write operations. This caused an `IndexError: Cannot insert operation at the end of a block that already has a terminator` when any kernel used a blocking `put()` on a shared stream.

**Reproducing Steps**:
1. Define a `df.region` with two kernels.
2. In the first kernel, call `some_stream.put(data)` (blocking put, not try_put).
3. In the second kernel, call `some_stream.get()`.
4. `df.build(top, target="simulator")` raises `IndexError`.

**How it is fixed**: Changed `ip=before_ip` to `ip=replace_ip` for the `tail_index_op = index_d.CastUOp(...)` call and all subsequent data-write operations in the `StreamPutOp` branch of the first (callee-arg) section of `_process_function_streams`. The write operations now correctly execute after the spin-while loop exits (FIFO has space), not inside the while's `before` condition block. The second (local-stream) section already had the correct `ip=replace_ip` at the equivalent position.

**Files and Lines affected**:
- `allo/backend/simulator.py`
  - In `_process_function_streams()`, first section (callee arg stream handling), `StreamPutOp` branch: `tail_index_op = index_d.CastUOp(..., ip=replace_ip)` (was `ip=before_ip`).

---

**Note:** An attempt was also made to modify `_build_top` in `allo/dataflow.py` (which generated a similar missing scalar resolution error on the outermost layer of `df.build`), but that ultimately caused an MLIR Segmentation Fault when emitting `AffineLoadOp` before regions returned. It was reverted; the current recommended workaround is explicitly typing scalar arguments in `df.region()` as 1D arrays `int32[1]` to side-step MLIR C++ segfaults.
