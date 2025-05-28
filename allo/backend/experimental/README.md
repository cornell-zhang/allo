<!--- Copyright Allo authors. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# Experimental MLIR-AIE Codegen
## Environment Setup
Please follow the [Getting Started](https://github.com/Xilinx/mlir-aie/tree/main?tab=readme-ov-file#getting-started-for-amd-ryzen-ai-on-linux) guide to install MLIR-AIE.

In **Step 3: Install IRON library, mlir-aie, and llvm-aie compilers from wheels**, under the section [Install IRON for AMD Ryzen™ AI AIE Application Development](https://github.com/Xilinx/mlir-aie/tree/main?tab=readme-ov-file#install-iron-for-amd-ryzen-ai-aie-application-development), please install version `v1.0` using the following commands:
```bash
# Install IRON library and mlir-aie from a wheel
python3 -m pip install mlir_aie -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/v1.0

# Install Peano from a llvm-aie wheel
python3 -m pip install https://github.com/Xilinx/llvm-aie/releases/download/nightly/llvm_aie-19.0.0.2025041501+b2a279c1-py3-none-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
```

Then, install Allo as usual:
```bash
git clone https://github.com/cornell-zhang/allo.git && cd allo
python3 -m pip install -v -e .
```

### Commands Used

Below are the exact commands to set up the environment:

1. create env and activate
   ```bash
   conda create -n allo python=3.12
   conda activate allo
   ```

2. install release 1.0
   ```bash
   # Install IRON library and mlir-aie from a wheel
   python3 -m pip install mlir_aie -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/v1.0
   # Install Peano from a llvm-aie wheel
   python3 -m pip install https://github.com/Xilinx/llvm-aie/releases/download/nightly/llvm_aie-19.0.0.2025041501+b2a279c1-py3-none-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
   ```

3. Clone the mlir-aie repository and checkout to the commit corresponding to release 1.0
   ```bash
   git clone https://github.com/Xilinx/mlir-aie.git
   cd mlir-aie
   git checkout 07320d6
   ```
4. Install
   ```bash
   # Install basic Python requirements 
   python3 -m pip install -r python/requirements.txt
   # Install the pre-commit hooks defined in .pre-commit-config.yaml
   pre-commit install
   # Install MLIR Python Extras 
   HOST_MLIR_PYTHON_PACKAGE_PREFIX=aie python3 -m pip install -r python/requirements_extras.txt
   # Install Torch for ML examples
   python3 -m pip install -r python/requirements_ml.txt
   ```

5. Setup environment and add tools to PATHs
   ```bash
   source utils/env_setup.sh
   ```

6. Clone the allo repository and install.
   - You may want to set up environment variables to use a custom CMake and LLVM build. For example, `export PATH=/opt/cmake-3.31.5-linux-x86_64/bin:/opt/llvm-project-19.x/build/bin:$PATH` and `export LLVM_BUILD_DIR=/opt/llvm-project-19.x/build`.
   ```bash
   git clone https://github.com/cornell-zhang/allo.git
   cd allo
   python3 -m pip install -v -e .
   ```

Do not forget to setup Vitis and XRT.

### Patches and Configuration
To use components from the [MLIR-AIE toolchain](https://github.com/Xilinx/mlir-aie) as libraries:

> ⚠️ **Note:** The instructions below are based on [MLIR-AIE release v1.0](https://github.com/Xilinx/mlir-aie/releases/tag/v1.0), which corresponds to commit [`07320d6`](https://github.com/Xilinx/mlir-aie/tree/07320d6831b17e4a4c436d48c3301a17c1e9f1cd).
> For compatibility, make sure to use this commit when copying the following components:

You can clone and checkout the specific commit with:

```bash
git clone https://github.com/Xilinx/mlir-aie.git
cd mlir-aie
git checkout 07320d6
```

- To use [external kernels](https://github.com/Xilinx/mlir-aie/tree/07320d6831b17e4a4c436d48c3301a17c1e9f1cd/aie_kernels) as an AIE kernel library, copy the directory to a desired location and set the environment variable:

  ```bash
  export MLIR_AIE_EXTERNAL_KERNEL_DIR=/your/copied/path/aie_kernels
  ```

- To use [runtime\_lib](https://github.com/Xilinx/mlir-aie/tree/07320d6831b17e4a4c436d48c3301a17c1e9f1cd/runtime_lib) for the host, copy it to a desired location and set the environment variable:

  ```bash
  export RUNTIME_LIB_DIR=/your/copied/path/runtime_lib
  ```

If you run into issues when using `aiecc.py`, such as:
```text
error: expected ')' at end of argument list
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
                                                          ^
``` 
You can fix this by modifying the `downgrade_ir_for_peano` function in:
```text
$MLIR_AIE_INSTALL_DIR/python/aie/compiler/aiecc/main.py
```

Update the function as follows:

**Before:**

```python
def downgrade_ir_for_peano(llvmir):
    llvmir = llvmir.replace("getelementptr inbounds nuw", "getelementptr inbounds")
    return llvmir
```

**After:**

```python
def downgrade_ir_for_peano(llvmir):
    llvmir = llvmir.replace("getelementptr inbounds nuw", "getelementptr inbounds")
    llvmir = llvmir.replace("captures(none)", "")
    return llvmir
```

## Usage

To enable the experimental MLIR-AIE codegen, set the following environment variable:

```bash
export USE_AIE_MLIR_BUILDER=1
```

Then, specify `"aie-mlir"` as the target in the `dataflow.build` function.

### Example
vector addition
```python
import os
import allo
from allo.ir.types import int32
import allo.dataflow as df
import numpy as np
from allo.memory import Layout

Ly = Layout("S0")


def _test_vector_scalar_add():
    # https://github.com/Xilinx/mlir-aie/tree/main/programming_examples/basic/vector_scalar_add
    Ty = int32
    M = 1024

    @df.region()
    def top():
        @df.kernel(mapping=[1])
        def core(A: Ty[M], B: Ty[M]):
            B[:] = allo.add(A, 1)

    A = np.random.randint(0, 100, M).astype(np.int32)
    if "MLIR_AIE_INSTALL_DIR" in os.environ:
        mod = df.build(top, target="aie-mlir")
        B = np.zeros(M).astype(np.int32)
        mod(A, B)
        np.testing.assert_allclose(B, A + 1)
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")
```

matrix multiplication
```python
import allo
from allo.ir.types import int32
import allo.dataflow as df
import numpy as np
from allo.memory import Layout

LyA = Layout("S0R")
LyB = Layout("RS1")
LyC = Layout("S0S1")


def _test_gemm_1D():
    Ty = int32
    M, N, K = 16, 16, 16
    P0 = 2

    @df.region()
    def top():
        @df.kernel(mapping=[P0])
        def gemm(A: Ty[M, K] @ LyA, B: Ty[K, N], C: Ty[M, N] @ LyA):
            C[:, :] = allo.matmul(A, B)

    mod = df.build(top, target="aie-mlir")
    A = np.random.randint(0, 64, (M, K)).astype(np.int32)
    B = np.random.randint(0, 64, (K, N)).astype(np.int32)
    C = np.zeros((M, N)).astype(np.int32)
    mod(A, B, C)
    np.testing.assert_allclose(C, A @ B, atol=1e-5)
    print("PASSED!")
```

producer consumer
```python
import os
import allo
from allo.ir.types import int32
import allo.dataflow as df
import numpy as np

Ty = int32
M, N, K = 16, 16, 16


@df.region()
def top():
    pipe = df.pipe(dtype=Ty, shape=(), depth=4)

    @df.kernel(mapping=[1])
    def producer(A: Ty[M, N]):
        for i, j in allo.grid(M, N):
            # load data
            out: Ty = A[i, j]
            # send data
            pipe.put(out)

    @df.kernel(mapping=[1])
    def consumer(B: Ty[M, N]):
        for i, j in allo.grid(M, N):
            # receive data
            data = pipe.get()
            # computation
            B[i, j] = data + 1


def test_producer_consumer():
    A = np.random.randint(0, 64, (M, K)).astype(np.int32)
    B = np.zeros((M, N), dtype=np.int32)

    if "MLIR_AIE_INSTALL_DIR" in os.environ:
        mod = df.build(top, target="aie-mlir")
        mod(A, B)
        np.testing.assert_allclose(A + 1, B, atol=1e-5)
        print("Passed!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")

```

large scale GEMM
```python
import allo
from allo.ir.types import int16, int32
import allo.dataflow as df
import numpy as np
from allo.memory import Layout

LyA = Layout("S0R")
LyB = Layout("RS1")
LyC = Layout("S0S1")


TyI, TyO = int16, int32
total_M, total_N, total_K = 128, 128, 512
M, N, K = 128, 128, 32


@df.region()
def top1():
    @df.kernel(mapping=[4, 4])
    def gemm(A: TyI[M, K] @ LyA, B: TyI[K, N] @ LyB, C: TyO[M, N] @ LyC):
        C[:, :] = allo.matmul(A, B)


@df.region()
def top2():
    @df.kernel(mapping=[2, 4])
    def core(A: TyO[M, N] @ LyC, B: TyO[M, N] @ LyC, C: TyO[M, N] @ LyC):
        C[:, :] = allo.add(A, B)


mod1 = df.build(top1, target="aie-mlir", project="top1.prj")
mod2 = df.build(top2, target="aie-mlir", project="top2.prj")

A = np.random.randint(0, 8, (total_M, total_K)).astype(np.int16)
B = np.random.randint(0, 8, (total_K, total_N)).astype(np.int16)
C_tmp = np.zeros((M, N)).astype(np.int32)
C = np.zeros((M, N)).astype(np.int32)

for i in range(total_K // K):
    tile_A = A[:, i * K : (i + 1) * K]
    tile_B = B[i * K : (i + 1) * K, :]
    mod1(tile_A, tile_B, C_tmp)
    mod2(C, C_tmp, C)

np.testing.assert_allclose(C, A @ B, atol=1e-5)
print("PASSED!")
```

### New Feature
#### Profiling
A new profiling feature has been added to help measure the performance of the module during execution. 

To enable profiling, use the `do_profile` flag in the `build` method in [`dataflow.py`](../../dataflow.py):
```python
def build(
    func,
    target="vitis_hls",
    mode="csim",
    project="top.prj",
    configs=None,
    wrap_io=True,
    opt_default=True,
    enable_tensor=False,
    profile=False,
    warmup=20,
    num_iters=100,
):
```

**New Parameters:**

- `profile` (`bool`): Set to `True` to enable profiling. When enabled, the module performs extra warm-up and test iterations.
- `warmup` (`int`): Number of initial iterations to warm up the system. These iterations are **excluded** from the timing measurements. Default is `20`.
- `num_iters` (`int`): Number of timed iterations used to compute execution time. Default is `100`.

##### Example
```python
import allo
from allo.ir.types import int16, int32, float32
import allo.dataflow as df
import numpy as np
from allo.memory import Layout

Ty = int16
M, N, K = 128, 128, 32
Pm, Pn, Pk = 4, 4, 1
Mt, Nt, Kt = M // Pm, N // Pn, K // Pk

LyA = Layout("S1S2")
LyB = Layout("S2S0")
LyC = Layout("S1S0")

@df.region()
def top1():
    @df.kernel(mapping=[Pk, Pm, Pn])
    def gemm(A: Ty[M, K] @ LyA, B: Ty[K, N] @ LyB, C: int32[M, N] @ LyC):
        C[:, :] = allo.matmul(A, B)

mod = df.build(
    top1,
    target="aie-mlir",
    profile=True,
    warmup=200,
    num_iters=1000,
)
A = np.random.randint(0, 32, (M, K)).astype(np.int16)
B = np.random.randint(0, 32, (K, N)).astype(np.int16)
C = np.zeros((M, N)).astype(np.int32)
tmp_C = np.zeros((M, N)).astype(np.int32)
mod(A, B, C)
```

### ⚠️ Note
Code that previously used `"aie"` as the target in the `dataflow.build` function may no longer work correctly in this environment.

This is mainly due to recent **syntax changes in `mlir-aie`**. For example, running:

```
tests/dataflow/aie/test_vector.py
```

may result in the following error:

```
Unable to parse module assembly: 
error: "-":44:30: expected SSA operand
```

This happens because the syntax of operations like the following has changed:

```
aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][1, 1, 1, 1024][0, 0, 0, 1]) {
  id = 0 : i64, issue_token = true, metadata = @in_shim_A
} : memref<1024xi32>
```
