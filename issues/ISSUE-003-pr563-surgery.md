# ISSUE-003: Create focused PR for hierarchical dataflow fixes (#561 / #565)

## Status: NEEDS-REVIEW (local branch `fix/hierarchical-dataflow-codegen` created — awaiting push approval)
## Upstream interaction: YES — close PR #563, open new PR (approval required)

## Context

PR #563 is open at cornell-zhang/allo#563 but its branch is `feature/mesh-accelerator`, which
carries the entire mesh accelerator feature (+6964/-137 lines). The maintainer asked for CI to be
fixed and the branch is also 1 commit behind origin/main. This PR is too large and unfocused for
upstream — it should only contain the three targeted bug-fix commits.

**Goal:** Replace PR #563 with a clean, minimal PR containing only the 3 commits that fix
issues #561 and #565, rebased cleanly onto `origin/main`.

## Commits to cherry-pick

These are the only commits relevant to upstream (the rest is our research work):

| Commit | Description | Files |
|--------|-------------|-------|
| `5e08403` | fix: simulator hierarchical deadlock — recursive OMP injection | `allo/backend/simulator.py` |
| `83905ea` | Fix HLS codegen for hierarchical dataflow regions (#561) — forward decls, dataflow pragma, name table reset | `allo/ir/builder.py`, `mlir/include/allo/Translation/EmitVivadoHLS.h`, `mlir/lib/Translation/EmitVivadoHLS.cpp` |
| `812d222` | Fix dataflow region nested kernel issues — `ASTContext.copy()` copies `global_op_cache` | `allo/ir/visitor.py` |

**Note:** `ecd0180` (earlier simulator fix) was already merged upstream as PR #562. Skip it.

## Fix procedure

```bash
# 1. Create a clean branch off current origin/main
git fetch origin
git checkout -b fix/hierarchical-dataflow-codegen origin/main

# 2. Cherry-pick the three fix commits (in order)
git cherry-pick 5e08403   # simulator recursive OMP
git cherry-pick 812d222   # ASTContext.copy global_op_cache
git cherry-pick 83905ea   # HLS codegen forward decls + pragma

# Note: if 83905ea has conflicts with 76130c6 (SPMW, already in origin/main),
# resolve manually — SPMW touched allo/ir/visitor.py, builder.py, and MLIR ops.

# 3. Run the targeted tests
./run_allo.sh python -m pytest tests/dataflow/test_hierachical_mesh.py -v

# 4. Push to fork (*** REQUIRES APPROVAL ***)
git push fork fix/hierarchical-dataflow-codegen

# 5. Close PR #563 with a comment explaining the replacement (*** REQUIRES APPROVAL ***)
# gh pr close 563 --repo cornell-zhang/allo --comment "Closing in favor of a focused PR on fix/hierarchical-dataflow-codegen"

# 6. Open new PR (*** REQUIRES APPROVAL ***)
# gh pr create --repo cornell-zhang/allo \
#   --head sunwookim028:fix/hierarchical-dataflow-codegen \
#   --base main \
#   --title "fix: hierarchical dataflow codegen and simulator deadlock (#561, #565)" \
#   --body "..."
```

## Conflict risk assessment

`76130c6` (SPMW) touched:
- `allo/ir/visitor.py` (also touched by `812d222`) — low conflict risk (different lines)
- `allo/ir/builder.py` (also touched by `83905ea`) — **medium conflict risk** — inspect carefully
- `mlir/lib/Dialect/AlloOps.cpp`, `AlloOps.td` — untouched by our commits

Resolve conflicts manually; do not use `--strategy-option=ours`.

## Completion notes (2026-04-10)

- Cherry-pick 1 (`5e08403`): clean, no conflicts.
- Cherry-pick 2 (`812d222`): auto-merged cleanly with SPMW visitor.py changes.
- Cherry-pick 3 (`83905ea`): conflict in `mlir/lib/Translation/EmitVivadoHLS.cpp` and
  `mlir/include/allo/Translation/EmitVivadoHLS.h`. The HEAD (origin/main + SPMW) had
  already refactored `emitFunctionSignature` to return `SmallVector<Value,8>` and added
  `emitFunctionDeclaration`. Resolved by:
  - Keeping HEAD's `SmallVector<Value,8>` return type (with portList construction added
    inside the function body).
  - Keeping HEAD's `emitFunctionDeclaration` helper.
  - Taking cherry-pick's third-pass forward-declaration loop and nameTable.clear() (the
    actual bug fix), adjusting the `os <<` to `"\n);\n\n"` to match the HEAD API.
  - Removing duplicate `portList` declaration in `emitFunction`.
  - Deduplicating the header declaration (removed the `void` version, kept SmallVector).
- Tests: simulator portion of `test_hierachical.py` PASSED. HLS build failed with
  "Disk quota exceeded" when creating project dir — infrastructure issue, not a code bug.
- Branch: `fix/hierarchical-dataflow-codegen` (3 commits ahead of origin/main, not pushed).

## What this unblocks

- Clean upstream PR for the hierarchical dataflow fix (closes #561 and #565)
- Removes the stale / oversized PR #563

## Files touched (cherry-picked)

- `allo/backend/simulator.py`
- `allo/ir/builder.py`
- `allo/ir/visitor.py`
- `mlir/include/allo/Translation/EmitVivadoHLS.h`
- `mlir/lib/Translation/EmitVivadoHLS.cpp`
