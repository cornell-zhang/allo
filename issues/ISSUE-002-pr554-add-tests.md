# ISSUE-002: Add test cases for PR #554

## Status: TODO
## Upstream interaction: YES — push to fork triggers CI re-run on PR #554 (approval required)

## Context

Maintainer `chhzh123` asked: "Can you add test cases for this?" in review of PR #554.
PR #554 fixes two bugs:

1. **`%alloc` strip** (`vitis.py`): `postprocess_hls_code()` now runs
   `re.sub(r"%(\w)", r"\1", hls_code)` to strip MLIR SSA `%`-prefixed names that would be
   illegal C++ identifiers in the csim wrapper.

2. **`ap_int` nanobind wrapper** (`ip.py`): `parse_cpp_function()` now handles `ap_int<N>` /
   `ap_uint<N>` type names; `resolve_nb_type()` maps them to `stdint` types (`int8_t`,
   `uint16_t`, etc.) in the generated nanobind wrapper, with a `reinterpret_cast` back to the
   HLS type in the call.

Neither fix requires Vitis HLS to run — both are pure Python string processing. Tests should be
unit tests in the existing test suite.

## Where to add tests

**Option A (preferred):** Add a new file `tests/test_ip_vitis_utils.py` with two test functions:
- `test_postprocess_strips_percent_alloc()` — feed a string with `%alloc` / `%alloc1` into
  `postprocess_hls_code()` and assert the `%` signs are removed.
- `test_resolve_nb_type_ap_int()` — call `resolve_nb_type("ap_int<8>")` etc. and assert
  expected `stdint` mapping.

**Option B:** Extend an existing test file that already imports from `allo.backend`.

Check existing test structure:
```bash
ls tests/test_backend*.py tests/test_*hls*.py 2>/dev/null
```

## Implementation notes

- `postprocess_hls_code` is in `allo/backend/vitis.py`
- `resolve_nb_type` and `parse_cpp_function` are in `allo/backend/ip.py`
- Both can be imported directly without HLS installed
- The test should NOT depend on Vitis HLS being present (so CI can run it without the HLS image)

## Fix procedure

```bash
# 1. Make sure on the branch (after ISSUE-001 is done)
git checkout fix/vhls-mlir-percent-alloc-csim

# 2. Write tests (see above)

# 3. Run locally
./run_allo.sh python -m pytest tests/test_ip_vitis_utils.py -v

# 4. Commit
git add tests/test_ip_vitis_utils.py
git commit -m "test: add unit tests for %alloc strip and ap_int nanobind resolve"

# 5. Push to fork (*** REQUIRES APPROVAL ***)
git push fork fix/vhls-mlir-percent-alloc-csim

# 6. Reply to maintainer comment on PR #554 noting tests added
# (*** REQUIRES APPROVAL ***)
```

## What this unblocks

- Addresses maintainer's remaining review request on PR #554
- Combined with ISSUE-001 (black fix), PR #554 should be merge-ready

## Files touched

- `tests/test_ip_vitis_utils.py` (new file)
