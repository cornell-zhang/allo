<!--- Copyright Allo authors. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# Project State

## Vision & Scope

This fork (`sunwookim028/allo`, branch `feature/mesh-accelerator`) extends the upstream
Allo compiler (`cornell-zhang/allo`) with:

1. **Non-blocking stream semantics** — `try_put`/`try_get`/`empty`/`full` FIFO primitives
   enabling valid-ready handshake protocols between dataflow kernels. Implemented in the
   MLIR dialect (AlloOps.td), Vitis HLS emitter, simulator backend.

2. **Tile-based hierarchical dataflow mesh** — a programming model and example architecture
   (Memory Tile + Compute Tiles) using decoupled control (NB streams) and burst data streams.
   Key example: `tests/dataflow/test_decoupled_mesh.py` (1-CT and 2×1 mesh).

3. **Upstream bug fixes** — hierarchical dataflow simulator/codegen fixes submitted as PRs.

**What we do NOT maintain:** Catapult HLS backend (removed; see `notes/ASIC_HLS_EXPLORATION.md`),
Tapa NB stream additions (removed). Primary synthesis target: Vitis HLS.

---

## Task Board

| ID | Title | Status | Upstream? | Blocked by |
|----|-------|--------|-----------|------------|
| [ISSUE-001](issues/ISSUE-001-pr554-black-format.md) | Black formatting on PR #554 branch | **DONE** | push to fork only | — |
| [ISSUE-002](issues/ISSUE-002-pr554-add-tests.md) | Add test cases for PR #554 | **DONE** | push + comment | ISSUE-001 |
| [ISSUE-003](issues/ISSUE-003-pr563-surgery.md) | Create focused PR replacing #563 | **NEEDS-REVIEW** (CI PASSED) | close PR + open new | ISSUE-004 |
| [ISSUE-004](issues/ISSUE-004-merge-upstream-spmw.md) | Merge upstream SPMW into feature branch | **DONE** | local only | — |
| [ISSUE-005](issues/ISSUE-005-u280-inner-product-overview.md) | U280 inner product — architecture & overview | **OPEN** | standalone RTL project | — |
| [ISSUE-006](issues/ISSUE-006-u280-rtl-implementation.md) | U280 inner product — RTL implementation (5 SV files) | **OPEN** | standalone RTL project | ISSUE-005 |
| [ISSUE-007](issues/ISSUE-007-u280-packaging-and-verify.md) | U280 inner product — packaging, build, sw_emu verify | **OPEN** | standalone RTL project | ISSUE-006 |
| [ISSUE-008](issues/ISSUE-008-fp16-hls-synthesis-verify.md) | Verify float16 arithmetic + exp synthesize in Vitis HLS | **DONE** | no (local verify; informs PR #578) | — |
| [ISSUE-009](issues/ISSUE-009-fp16-builder-scalar-exp.md) | Fix scalar exp dispatch in builder.py for float16 | **DONE** | yes (PR against upstream) | ISSUE-008 |
| [ISSUE-010](issues/ISSUE-010-fp16-emitter-exp-half.md) | Fix exp(half) emitter in EmitVivadoHLS.cpp | **DONE** | yes (follow-up to PR #578) | ISSUE-008 |

---

## Dependency Graph

```
ISSUE-001  ──►  ISSUE-002  ──►  (PR #554 merge-ready, await maintainer)
                                          │
ISSUE-004  ──►  ISSUE-003  ──►  (PR #577 opened, replaces #563, CI running)

ISSUE-005 (arch) ──►  ISSUE-006 (RTL) ──►  ISSUE-007 (package+verify)

ISSUE-008 (fp16 csyn verify) ──►  ISSUE-009 (builder.py fix, upstream PR)
                              ──►  ISSUE-010 (emitter fix, conditional on Test B fail)
                       [standalone U280 inner product RTL kernel project]
```

- **ISSUE-001 → ISSUE-002**: Black format fix must land on the branch before test push so CI
  runs clean.
- **ISSUE-004 → ISSUE-003**: The cherry-pick in ISSUE-003 is easier / less conflict-prone once
  we know SPMW merges cleanly. Also confirms the commit SHA of `origin/main` the new PR targets.
- ISSUE-001 and ISSUE-004 are **independent** and can proceed in parallel.

---

## Approval Gates

Any step that touches GitHub upstream requires explicit user approval **before** execution:

- Pushing to `fork` remote (triggers CI on open PRs)
- `gh pr close` / `gh pr create` / `gh pr comment`
- `git push fork <branch>`

Steps that are purely local (edit, commit, local merge, run tests) do **not** require approval.

---

## Upstream Watch

- **PR #574** (`fix/dataflow-kernel-ordering`, zzzDavid, open) — touches `allo/dataflow.py`;
  our `fix/hierarchical-dataflow-codegen` also touches it. Rebase needed if #574 lands first.
- **PR #570** (`spmw-builder-tmp`, Fangtangtang, open) — touches `allo/ir/types.py`;
  our `stateful = Stateful` alias is there. Rebase needed if #570 lands first.
- Note: upstream already had a Catapult backend (PR #543, zzzDavid, merged 2026-02-05).
  Our Catapult work was modifications on top of that. Both are now removed from this branch.

---

## Completed This Cycle (2026-04-13, continued)

- ISSUE-001: PR #554 black fmt — DONE
- ISSUE-002: PR #554 tests — DONE
- ISSUE-003: PR #577 opened (replaces #563) — NEEDS-REVIEW (CI PASSED — awaiting maintainer)
- ISSUE-004: SPMW merge — DONE
- Catapult/Tapa removal — DONE (commit `89339f0`)
- `stateful` alias removed; all files updated to `Stateful` — DONE (commit `e665bbc`)
- Project structure: `CLAUDE.md` created, root `STATE.md` + `notes/` + `issues/` restructured — DONE (commit `039e163`)
- PR #554 CI: PASSED; PR #577 CI: PASSED
- ISSUE-008: fp16 HLS synthesis verified — Test A PASS (half synthesizes, LUT=3986, FF=4233, 411 MHz), Test B FAIL (exp(half) ambiguous in hls_math.h → ISSUE-010) — DONE
- ISSUE-009: builder.py scalar math dispatch now handles F16Type/BF16Type — DONE (fixes allo.exp(float16) Python→MLIR lowering)
- ISSUE-010: EmitVivadoHLS.cpp math unary ops now emit hls::exp etc. for Float16Type — DONE; also fixed missing TapaHLS emitStreamTry*/Empty/Full implementations; both fp16 synth tests PASS (arith: LUT=3986/FF=4233; exp: LUT=2668/FF=2576)

---

## Out-of-Scope (Deferred)

- Catapult HLS backend: removed from branch; see `notes/ASIC_HLS_EXPLORATION.md`
- Tapa NB stream support: removed; Vitis HLS is the validated primary target
- 2×2 decoupled mesh, credit-based flow control: future work on `feature/mesh-accelerator`
- CIRCT backend: future high-priority item once upstream PRs are resolved

---

## U280 Inner Product RTL Kernel (ISSUE-005 to 007)

Issues moved to standalone project: `u280_inner_product/issues/`
Entry point for a fresh agent: `u280_inner_product/AGENT.md`
Project is self-contained; move `u280_inner_product/` outside Allo to use independently.

Harness skeleton: `u280_producer_consumer_hw.prj/` — generated by
`tests/u280_hw_deploy.py --codegen-only`. Copied into `u280_inner_product/baseline/` (complete
harness) and `u280_inner_product/harness/` (xcl2.* + utils.mk only). Run
`./run_allo.sh python tests/u280_hw_deploy.py --codegen-only` to regenerate if missing.
