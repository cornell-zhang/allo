# Building
- Always run `conda activate allo` before building or running tests
- Run `pip install -v -e .` to build the full project (includes MLIR/C++ backend)
- Read `docs/source/dive/frontend_syntax.rst` for comprehensive Allo frontend syntax reference
- Read `docs/source/dive/dataflow.rst` for the dataflow programming model (regions, kernels, streams)

# Testing
- Run `bash scripts/lint/task_lint.sh` for formatting checks
- Run `python3 -m pytest --ignore=tests/dataflow tests -v` for tests
  - Prefer running a single test file instead of the full suite (full suite is slow)
  - Use only software simulators (`target="llvm"` or `target="simulator"`)
  - If Vitis HLS tests are needed, ask the user to run them manually

# Code style
- Make small, targeted diffs rather than large refactors, and always be concise
- Prefer general solutions instead of one-off `if/else` patches
- Place Python frontend code in `allo/`
- Place MLIR dialects and passes code in `mlir/`
- Add tests and documentation for new features in `tests/` and `docs/`

# Don'ts
- Do not modify repository structure without approval
- Do not install system packages without explicit user confirmation

# Troubleshooting & Known Issues
- **"Fail to resolve the expression as symbolic expression" in Dataflow**: When using stream arrays (e.g., `gemm_in_A[m]`), the index `m` must be a compile-time constant (like `df.get_pid()`) or statically unrollable. Using a dynamic runtime loop index or variable will cause this type-inference failure. Manually unroll the loops or use literal constants where possible.
- **"AttributeError: 'ASTContext' object has no attribute 'global_op_cache'"**: This occurs when compiling stateful variables in nested kernels inside `df.region()`. Fixed upstream; ensure `ASTContext.copy()` properly preserves `self.global_op_cache`. Documented in `ALLO_CHANGE.md`.
- **"func.call op operand type mismatch: expected operand type 'i32', but provided 'memref<i32>'"**: When passing scalars into a `df.region()` top-level function, MLIR's `_build_top` lowers the arguments to memrefs, causing a crash when passed to inner scalar-expecting implementations. **Workaround**: Always type scalar arguments in `df.region()` and `df.kernel()` as 1-element arrays, e.g., `size: int32[1]` instead of `int32`, and access via `size[0]`.
- **Duplicate annotated variable names cause silent hangs**: In Allo, `x: int32 = expr` is an annotated assignment that creates a new buffer. If the same variable name is annotated twice in the same kernel (e.g., `addr1: int32 = ...` in two separate `if` branches), Allo may silently hang during compilation or simulation. **Fix**: Use unique names for each branch (e.g., `g_addr1` / `v_addr1`).
- **Dataflow deadlock from conditional stream usage**: In `df.region()`, all inner kernels execute exactly once per invocation. If one kernel conditionally `.put()`s to a stream but another kernel unconditionally `.get()`s from the same stream, the reader blocks forever when no data is sent. **Fix**: Use explicit enable streams (`en: Stream[int32, depth]`) that are always `.put()` by the producer and always `.get()` by the consumer. Guard the actual data stream `.get()` behind `if en == 1:`.
- **Multi-instruction loop pattern**: To execute multiple instructions per `CTRL_RUN`, all kernels must iterate a fixed `IMEM_SIZE` times using `for pc in range(IMEM_SIZE):`. The controller broadcasts enable signals each iteration; compute kernels conditionally execute or skip based on the enable value.
- **Multiple calls to inner df.region cause `@stateful` global redefinition**: If a `df.kernel` calls the same `df.region` from multiple branches (e.g., different `if/elif` arms), the MLIR builder emits the region's `@stateful` global declarations once per call site, causing `error: redefinition of symbol`. **Fix**: Restructure the kernel so there is exactly ONE call site to each inner region — set up arguments in branches, then call unconditionally at the end. Use a `CTRL_NOP` sentinel value if you need a no-op path.
- **Nested OMP parallelism deadlock**: When peer `df.kernel`s inside a `df.region` call sub-`df.region`s, nested OMP parallel sections deadlock because OpenMP serializes nested parallelism by default. **Fix**: The simulator backend now auto-sets `OMP_MAX_ACTIVE_LEVELS=4`. If running tests manually, set this env var.
- **All `@df.kernel` names must be globally unique**: Across the entire module, every `@df.kernel` function must have a unique Python name. Duplicate names cause MLIR symbol collisions. Use suffixes like `_t1`, `_p0`, `_p1` to distinguish per-tile instances.

# Key File Pointers (2x2 Mesh Accelerator)
- **Implementation**: [test_mesh_accelerator.py](tests/dataflow/test_mesh_accelerator.py)
- **Architecture Plan**: [PLAN.md](tests/dataflow/PLAN.md)
- **ISA Document**: [ISA.md](tests/dataflow/ISA.md)
- **Progress Log**: [PROGRESS.md](tests/dataflow/PROGRESS.md)
- **Compiler Changes**: [ALLO_CHANGE.md](ALLO_CHANGE.md)
- **Simulator Fix**: [simulator.py](allo/backend/simulator.py) — OMP injection for nested regions
- **Upstream Issue**: [cornell-zhang/allo#561](https://github.com/cornell-zhang/allo/issues/561) — HLS codegen bugs for nested regions

