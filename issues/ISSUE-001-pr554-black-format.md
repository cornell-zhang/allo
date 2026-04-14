<!--- Copyright Allo authors. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# ISSUE-001: Fix black formatting on PR #554 branch

## Status: TODO
## Upstream interaction: NO (local fix + push to fork only)

## Context

PR #554 ("fix: strip MLIR %alloc names from VHLS csim output and handle ap_int in nanobind
wrapper") is open at cornell-zhang/allo#554. Its only CI failure is a `black` formatting check.
No test/logic changes needed here — pure cosmetic.

**Branch:** `fix/vhls-mlir-percent-alloc-csim` (lives on `fork` remote: `sunwookim028/allo`)
**Failing job:** `cornell-zhang/allo` Actions run 22434564422 — `black 24.8.0` lint step

## Exact failures reported by black

### allo/backend/ip.py (around line 161 and ~185)

Two `enumerate(zip(...))` calls are split across 3 lines with a trailing paren. Black collapses
them to one line:

```diff
-        for i, ((arg_type, arg_shape), nb_type) in enumerate(
-            zip(self.args, nb_types)
-        ):
+        for i, ((arg_type, arg_shape), nb_type) in enumerate(zip(self.args, nb_types)):
```

### allo/backend/vitis.py (around line 384)

Single-quoted regex strings; black enforces double quotes:

```diff
-    hls_code = re.sub(r'%(\w)', r'\1', hls_code)
+    hls_code = re.sub(r"%(\w)", r"\1", hls_code)
```

## Fix procedure

```bash
# 1. Checkout the branch from fork
git checkout -b fix/vhls-mlir-percent-alloc-csim fork/fix/vhls-mlir-percent-alloc-csim

# 2. Run black on the two files (conda env has black installed)
./run_allo.sh black allo/backend/ip.py allo/backend/vitis.py

# 3. Verify nothing else changed
git diff --stat

# 4. Commit
git add allo/backend/ip.py allo/backend/vitis.py
git commit -m "style: apply black formatting to ip.py and vitis.py"

# 5. Push to fork (triggers CI on upstream PR #554)
# *** REQUIRES APPROVAL before step 5 ***
git push fork fix/vhls-mlir-percent-alloc-csim
```

## What this unblocks

- Clears the only CI failure on PR #554.
- Maintainer's remaining request (test cases) is tracked in ISSUE-002.

## Files touched

- `allo/backend/ip.py`
- `allo/backend/vitis.py`
