# ISSUE-004: Merge upstream SPMW commit into feature/mesh-accelerator

## Status: DONE
## Upstream interaction: NO (local merge only; no push until reviewed)

## Context

Upstream `origin/main` has one commit not yet in `feature/mesh-accelerator`:

```
76130c6  [MLIR][Dataflow] Add Allo operations for SPMW and setup support (#555)
```

This is a significant MLIR-side addition (653 insertions): new ops in `AlloOps.td`,
`LowerMemCopyOps.cpp`, updated `infer.py`, `visitor.py`, `builder.py`, `types.py`, `utils.py`,
and `pyproject.toml`. It is **unrelated to our mesh work** but shares several files.

**Impact on our branch:** The SPMW commit touches files we also modified:
- `allo/ir/visitor.py` — we added `global_op_cache` copy in `ASTContext.copy()`; SPMW adds
  `SPMW`-related visitor logic
- `allo/ir/builder.py` — we added 0-D MemRef unwrapping; SPMW adds SPMW builder support
- `mlir/include/allo/Dialect/AlloOps.td` — we have not touched this; SPMW adds new ops
- `mlir/lib/Translation/EmitVivadoHLS.cpp` — we added forward decls / non-blocking streams;
  SPMW may add SPMW emission (verify)

## Fix procedure

```bash
# 1. Check for conflicts first (dry run)
git -C . merge --no-commit --no-ff origin/main 2>&1

# If conflicts:
# - allo/ir/visitor.py: keep both sets of changes (our global_op_cache + SPMW logic)
# - allo/ir/builder.py: keep both sets of changes (our unwrapping + SPMW support)
# - Any MLIR C++ file: inspect carefully; our changes are in EmitVivadoHLS.cpp,
#   EmitCatapultHLS.cpp, EmitTapaHLS.cpp — SPMW likely only touches AlloOps.cpp/td

# 2. Perform the merge
git checkout feature/mesh-accelerator
git merge origin/main -m "Merge upstream main (SPMW ops #555) into feature/mesh-accelerator"

# 3. Rebuild the MLIR C++ layer (required if .td or .cpp files changed)
make -C mlir/build-rhel8 -j4   # on zhang-21
# or: touch mlir/build/build.ninja && conda run -n allo bash -c 'cd mlir/build && ninja -j4'

# 4. Run the core test suite to verify no regressions
./run_allo.sh python -m pytest tests/dataflow/test_stream_ops_sim.py \
    tests/dataflow/test_stream_nb_simple.py \
    tests/dataflow/test_decoupled_mesh.py -v

# 5. Keep on local branch only — do not push until ISSUE-003 is resolved
```

## Conflict resolution guidance

For `allo/ir/visitor.py`:
- Our change: `ASTContext.copy()` copies `global_op_cache` (two lines, `812d222`)
- SPMW change: adds SPMW-related visitor dispatch; different methods
- Resolution: accept both

For `allo/ir/builder.py`:
- Our change: 0-D MemRef unwrapping for scalar args in `func.call` (`83905ea`)
- SPMW change: SPMW builder hooks
- Resolution: accept both; check that our scalar unwrapping still applies after SPMW changes

## What this unblocks

- Keeps `feature/mesh-accelerator` current with upstream (required before any future PR)
- Required context for ISSUE-003 (clean cherry-pick branch must be based on post-SPMW main)

## Files likely to need conflict resolution

- `allo/ir/visitor.py`
- `allo/ir/builder.py`

---

## Resolution (2026-04-10)

### Conflicts encountered

Only one conflict in `mlir/include/allo/Dialect/AlloOps.td`:
- Our branch added: `StreamTryGetOp`, `StreamTryPutOp`, `StreamFullOp`, `StreamEmptyOp`
- Upstream SPMW added (in the same location): `StreamGlobalOp`, `GlobalStreamGetOp`, `GlobalStreamPutOp`
- Also: `YieldOp.ParentOneOf` changed from `["AndOp, OrOp"]` to `["AndOp", "OrOp", "GridMapOp"]`

Resolution: kept both sets (NB stream ops + SPMW global stream ops, plus the `GridMapOp` YieldOp change).

`allo/ir/visitor.py` and `allo/ir/builder.py` merged automatically (no manual intervention needed).

### Post-merge fix required

The SPMW commit renamed `def stateful(dtype)` to `class Stateful` in `allo/ir/types.py`.
This broke `test_decoupled_mesh.py` which imports `stateful` (lowercase).
Fix: added `stateful = Stateful` backward-compatibility alias in `allo/ir/types.py`.

### Commits

- Merge commit: `f8fa50c` — "Merge upstream main (SPMW ops #555) into feature/mesh-accelerator"
- Fix commit: `5595fab` — "fix: add stateful backward-compat alias after SPMW upstream merge"

### Test results

All 7 core tests passed after the fix:
- `test_stream_ops_sim.py::test_stream_nb_sim` PASSED
- `test_stream_nb_simple.py::test_try_put_try_get_sim` PASSED
- `test_stream_nb_simple.py::test_empty_full_sim` PASSED
- `test_stream_nb_simple.py::test_nb_ops_hls_codegen` PASSED
- `test_stream_nb_simple.py::test_nb_ops_tapa_codegen` PASSED
- `test_decoupled_mesh.py::test_decoupled_message_passing` PASSED
- `test_decoupled_mesh.py::test_decoupled_2x1_mesh` PASSED
