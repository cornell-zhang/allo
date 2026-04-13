# Environment Setup — Per-Server, Per-Use-Case

## Server Inventory

| Server | Hostname | OS / glibc | Notes |
|--------|----------|-----------|-------|
| Primary dev (FPGA) | `brg-zhang-xcel.ece.cornell.edu` | Ubuntu / glibc 2.35+ | Vitis HLS 2023.2, U280, GCC system ≥ 9 |
| Secondary dev (ASIC) | `zhang-21.ece.cornell.edu` | RHEL 8.10 / glibc **2.28** | Catapult 2024.x, no GCC-13, read-only conda env |

---

## Conda Environment: `allo`

```bash
conda activate allo    # sets LLVM_BUILD_DIR, PYTHONPATH, PATH
```

The activate script (`/work/shared/users/phd/sk3463/envs/miniconda3/envs/allo/etc/conda/activate.d/env_vars.sh`) sets:

| Variable | `brg-zhang-xcel` value | `zhang-21` value |
|----------|----------------------|-----------------|
| `LLVM_BUILD_DIR` | `.../llvm-project-main/build` | `.../llvm-project-main/build-rhel8` |
| `PYTHONPATH` | `build/tools/mlir/python_packages/mlir_core` | `build-rhel8/tools/mlir/...` |

---

## CRITICAL: `LD_LIBRARY_PATH` on `zhang-21` (RHEL 8)

**Problem**: `libAlloMLIRAggregateCAPI.so.22.0git` requires `GLIBCXX_3.4.30` (GCC 13). The system
`libstdc++.so.6` on RHEL 8 only has GCC 8 (`GLIBCXX_3.4.25`). The conda env ships a newer
`libstdc++.so.6` that satisfies this, but it won't be found unless `LD_LIBRARY_PATH` is set.

**Required for every session on `zhang-21`:**

```bash
export LD_LIBRARY_PATH="/work/shared/users/phd/sk3463/envs/miniconda3/envs/allo/lib:$LD_LIBRARY_PATH"
```

**Permanent fix** — add to `~/.bashrc` or `~/.bash_profile`:

```bash
# Allo MLIR: use conda libstdc++ on RHEL 8
if [[ "$(hostname)" != "brg-zhang-xcel.ece.cornell.edu" ]]; then
    CONDA_ENV_LIB="/work/shared/users/phd/sk3463/envs/miniconda3/envs/allo/lib"
    [[ ":$LD_LIBRARY_PATH:" != *":$CONDA_ENV_LIB:"* ]] && \
        export LD_LIBRARY_PATH="$CONDA_ENV_LIB:$LD_LIBRARY_PATH"
fi
```

**Wrapper script** (alternative) — use `./run_allo.sh <cmd>` instead of `conda run -n allo <cmd>`:

```bash
./run_allo.sh python -m pytest tests/dataflow/...
```

See `run_allo.sh` in the project root.

---

## Running Tests

```bash
# From project root, with correct LD_LIBRARY_PATH:
export LD_LIBRARY_PATH="/work/shared/users/phd/sk3463/envs/miniconda3/envs/allo/lib:$LD_LIBRARY_PATH"

# Simulator tests (no HLS tool required):
conda run -n allo python -m pytest \
    tests/dataflow/test_stream_ops_sim.py \
    tests/dataflow/test_stream_nb_simple.py \
    tests/dataflow/test_decoupled_mesh.py -v

# HLS codegen tests (Vitis HLS required; brg-zhang-xcel only):
conda run -n allo python -m pytest tests/dataflow/test_stream_ops_hls.py -v
```

---

## Catapult HLS (`zhang-21` or any server with Catapult)

### Load Catapult

```bash
module load mentor-Catapult_synthesis_10.5a
# Sets MGC_HOME, updates PATH
```

Or auto-detected from `/opt/siemens/catapult/{2024.2, 2024.1_2-1117371}/`.

### Generate Kernel Code (no Catapult required)

```bash
conda run -n allo python tests/dataflow/catapult_synth_decoupled_2x1.py --mode codegen
```

### Run Synthesis (Catapult required)

```bash
conda run -n allo python tests/dataflow/catapult_synth_decoupled_2x1.py --mode csyn
```

### Extract PPA Report (Catapult required)

```bash
conda run -n allo python tests/dataflow/catapult_synth_decoupled_2x1.py --mode ppa
```

---

## C++ Rebuild Procedure (MLIR Extension)

**Do not run cmake directly** — the conda cmake (4.1.2) fails to regenerate `build.ninja` due to
an RPATH issue.

### Normal rebuild (existing build dir, `build/` or `build-rhel8/`):

```bash
touch /work/shared/users/phd/sk3463/projects/allo/mlir/build/build.ninja  # prevents re-config
conda run -n allo bash -c 'cd /work/shared/users/phd/sk3463/projects/allo/mlir/build && ninja -j4'
```

### Rebuild for RHEL 8 (zhang-21) using `build-rhel8/` LLVM + conda libstdc++:

```bash
# Configure (system cmake 3.31.5, system GCC 8, conda libstdc++ for linking)
CONDA_LIB=/work/shared/users/phd/sk3463/envs/miniconda3/envs/allo/lib
/work/shared/common/cmake-3.31.5-linux-x86_64/bin/cmake \
  -S /work/shared/users/phd/sk3463/projects/allo/mlir \
  -B /work/shared/users/phd/sk3463/projects/allo/mlir/build-rhel8 \
  -DLLVM_DIR=/work/shared/common/llvm-project-main/build-rhel8/lib/cmake/llvm \
  -DMLIR_DIR=/work/shared/common/llvm-project-main/build-rhel8/lib/cmake/mlir \
  -DPython3_EXECUTABLE=/work/shared/users/phd/sk3463/envs/miniconda3/envs/allo/bin/python3.12 \
  -DPython_EXECUTABLE=/work/shared/users/phd/sk3463/envs/miniconda3/envs/allo/bin/python3.12 \
  -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_C_COMPILER=/usr/bin/gcc \
  "-DCMAKE_EXE_LINKER_FLAGS=-L${CONDA_LIB} -Wl,-rpath,${CONDA_LIB}" \
  "-DCMAKE_SHARED_LINKER_FLAGS=-L${CONDA_LIB} -Wl,-rpath,${CONDA_LIB}"

# Build
make -C /work/shared/users/phd/sk3463/projects/allo/mlir/build-rhel8 -j4

# Install: copy built .so files to the allo package
SRC=/work/shared/users/phd/sk3463/projects/allo/mlir/build-rhel8/tools/allo/_mlir/_mlir_libs
DST=/work/shared/users/phd/sk3463/projects/allo/allo/_mlir/_mlir_libs
for f in "$SRC"/*.so; do cp "$f" "$DST/$(basename $f)"; done

# Also copy the aggregate CAPI library
cp /work/shared/users/phd/sk3463/projects/allo/mlir/build-rhel8/tools/allo/_mlir/libAlloMLIRAggregateCAPI.so.22.0git \
   /work/shared/users/phd/sk3463/projects/allo/mlir/build/tools/allo/_mlir/
```

---

## Key File Paths

| Item | Path |
|------|------|
| Allo project | `/work/shared/users/phd/sk3463/projects/allo/` |
| Conda env | `/work/shared/users/phd/sk3463/envs/miniconda3/envs/allo/` |
| LLVM build (brg-zhang-xcel) | `/work/shared/common/llvm-project-main/build/` |
| LLVM build (zhang-21, RHEL 8) | `/work/shared/common/llvm-project-main/build-rhel8/` |
| System cmake 3.31.5 | `/work/shared/common/cmake-3.31.5-linux-x86_64/bin/cmake` |
| Catapult 2024.2 | `/opt/siemens/catapult/2024.2/` |
| Catapult 2024.1 | `/opt/siemens/catapult/2024.1_2-1117371/` |
| GLIBC compat shim | `glibc_compat/libglibc_compat.so` (built but not needed after rebuild) |
| HLS projects | `hls_projects/` |

---

## Known Incompatibilities

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| `GLIBC_2.34` error on zhang-21 | `.so` built on brg-zhang-xcel with glibc 2.35, RHEL 8 has 2.28 | Rebuild with `build-rhel8` LLVM (done) |
| `GLIBCXX_3.4.26` error | `libAlloMLIRAggregateCAPI.so` needs GCC 13 libstdc++ | Set `LD_LIBRARY_PATH` to conda's lib (GCC 13 libstdc++) |
| `cmake` fails to regenerate `build.ninja` | conda cmake 4.1.2 has RPATH bug on RHEL 8 | Use `touch build.ninja` before ninja; use system cmake 3.31.5 for new builds |
| Catapult module not on zhang-21 | `mentor-Catapult_synthesis_10.5a` not installed | Use `/opt/siemens/catapult/` path directly or run on a server with Catapult |
| `Option 'fast' already exists` | LLVM loaded twice from `build/` + `build-rhel8` | Use all `.so` files from same LLVM build (done) |
