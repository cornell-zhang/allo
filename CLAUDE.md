<!--- Copyright Allo authors. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

## This fork's project management conventions

This fork (`sunwookim028/allo`, branch `feature/mesh-accelerator`) extends upstream Allo with
non-blocking stream primitives, a tile-based hierarchical dataflow mesh, and simulator/codegen
fixes. The following conventions govern how agents should navigate and update the project.

### notes/ — Modular knowledge files

`notes/` contains standalone reference documents covering architecture decisions, environment
setup, synthesis results, and exploration work. Examples: `ENVIRONMENT.md`, `HLS_SYNTH_REPORT.md`,
`ASIC_HLS_EXPLORATION.md`, `ALLO_CHANGE.md`.

**Agent rule**: When you discover new knowledge (a synthesis result, a bug root-cause, an env
quirk), find the relevant note and update it, or create a new note if none fits. Keep notes
factual and self-contained so any agent can read one note and understand the topic.

### issues/ — Self-contained task files

`issues/` contains one `.md` file per task (`ISSUE-NNN-short-title.md`). Each file describes
the problem, the plan, acceptance criteria, and status (`OPEN` / `IN-PROGRESS` / `DONE`).

**Agent rule**: When you complete work that resolves a task, mark the issue `DONE` and update
any relevant notes in the same commit. Never close an issue file without also checking whether
`STATE.md` needs updating.

### STATE.md — Project dashboard (root)

`STATE.md` at the repo root is the single source of truth for project vision, the task board
(status table of all ISSUE-NNN items), dependency graph, and upstream watch items.

**Agent rule**: After completing a task or merging upstream changes, update the task board row
in `STATE.md` in the same commit. Keep the "Completed This Cycle" section current.

### Commit hygiene

When a commit resolves a task tracked in `issues/`:

1. Update the issue file status to `DONE`.
2. Update the `STATE.md` task board row.
3. Update any affected note in `notes/`.
4. Include all three changes in the same commit.

Any step that touches GitHub upstream (push, `gh pr create/close/comment`) requires **explicit
user approval before execution**.

---

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

# Code Quality
- All implementation must follow the project's and relevant community's established practices
- Web-search idiomatic patterns before writing non-trivial code if not 100% confident
- No ad-hoc patching: check how similar features are done in the same file/module first
- This applies to all upstream PRs, compiler changes, MLIR passes, and HLS backend code

# Filesystem
- `/work/shared/users/phd/sk3463/` — NFS home for source files and docs; **quota-limited**
- `/scratch/sk3463/` — local scratch, ~1.8 TB free; use for HLS project dirs, build artifacts, large outputs

# Troubleshooting & Known Issues
- **"Fail to resolve the expression as symbolic expression" in Dataflow**: When using stream arrays (e.g., `gemm_in_A[m]`), the index `m` must be a compile-time constant (like `df.get_pid()`) or statically unrollable. Using a dynamic runtime loop index or variable will cause this type-inference failure. Manually unroll the loops or use literal constants where possible.
- **"AttributeError: 'ASTContext' object has no attribute 'global_op_cache'"**: This occurs when compiling stateful variables in nested kernels inside `df.region()`. Fixed upstream; ensure `ASTContext.copy()` properly preserves `self.global_op_cache`. Documented in `notes/ALLO_CHANGE.md`.
- **"func.call op operand type mismatch: expected operand type 'i32', but provided 'memref<i32>'"**: When passing scalars into a `df.region()` top-level function, MLIR's `_build_top` lowers the arguments to memrefs, causing a crash when passed to inner scalar-expecting implementations. **Workaround**: Always type scalar arguments in `df.region()` and `df.kernel()` as 1-element arrays, e.g., `size: int32[1]` instead of `int32`, and access via `size[0]`.
- **Duplicate annotated variable names cause silent hangs**: In Allo, `x: int32 = expr` is an annotated assignment that creates a new buffer. If the same variable name is annotated twice in the same kernel (e.g., `addr1: int32 = ...` in two separate `if` branches), Allo may silently hang during compilation or simulation. **Fix**: Use unique names for each branch (e.g., `g_addr1` / `v_addr1`).
- **Dataflow deadlock from conditional stream usage**: In `df.region()`, all inner kernels execute exactly once per invocation. If one kernel conditionally `.put()`s to a stream but another kernel unconditionally `.get()`s from the same stream, the reader blocks forever when no data is sent. **Fix**: Use explicit enable streams (`en: Stream[int32, depth]`) that are always `.put()` by the producer and always `.get()` by the consumer. Guard the actual data stream `.get()` behind `if en == 1:`.
- **Multi-instruction loop pattern**: To execute multiple instructions per `CTRL_RUN`, all kernels must iterate a fixed `IMEM_SIZE` times using `for pc in range(IMEM_SIZE):`. The controller broadcasts enable signals each iteration; compute kernels conditionally execute or skip based on the enable value.
- **Multiple calls to inner df.region cause `@stateful` global redefinition**: If a `df.kernel` calls the same `df.region` from multiple branches (e.g., different `if/elif` arms), the MLIR builder emits the region's `@stateful` global declarations once per call site, causing `error: redefinition of symbol`. **Fix**: Restructure the kernel so there is exactly ONE call site to each inner region — set up arguments in branches, then call unconditionally at the end. Use a `CTRL_NOP` sentinel value if you need a no-op path.
- **Nested OMP parallelism deadlock**: When peer `df.kernel`s inside a `df.region` call sub-`df.region`s, nested OMP parallel sections deadlock because OpenMP serializes nested parallelism by default. **Fix**: The simulator backend now auto-sets `OMP_MAX_ACTIVE_LEVELS=4`. If running tests manually, set this env var.
- **All `@df.kernel` names must be globally unique**: Across the entire module, every `@df.kernel` function must have a unique Python name. Duplicate kernel names cause MLIR symbol collisions. Use suffixes like `_t1`, `_p0`, `_p1` to distinguish per-tile instances.

# Key File Pointers

- **Non-blocking stream tests**: [`tests/dataflow/test_stream_nb_simple.py`](tests/dataflow/test_stream_nb_simple.py)
- **Decoupled mesh (1-CT and 2×1)**: [`tests/dataflow/test_decoupled_mesh.py`](tests/dataflow/test_decoupled_mesh.py)
- **Blocking mesh reference**: [`tests/dataflow/test_hierachical_mesh.py`](tests/dataflow/test_hierachical_mesh.py)
- **Simulator backend**: [`allo/backend/simulator.py`](allo/backend/simulator.py)
- **Compiler changes log**: [`notes/ALLO_CHANGE.md`](notes/ALLO_CHANGE.md)
- **Project state (task board)**: [`STATE.md`](STATE.md)
- **Task issues**: [`issues/`](issues/)
- **Knowledge notes**: [`notes/`](notes/)
