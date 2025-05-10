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

### Patches and Configuration
We rely on components from the [MLIR-AIE toolchain](https://github.com/Xilinx/mlir-aie) as libraries:

- To use [external kernels](https://github.com/Xilinx/mlir-aie/tree/ea9b4dfe7ea91f09c5c29c4d51ca74baea2dc4aa/aie_kernels) as an AIE kernel library, copy the directory to a desired location and set the environment variable:

  ```bash
  export MLIR_AIE_EXTERNAL_KERNEL_DIR=/your/copied/path/aie_kernels
  ```

- To use [runtime\_lib](https://github.com/Xilinx/mlir-aie/tree/ea9b4dfe7ea91f09c5c29c4d51ca74baea2dc4aa/runtime_lib) for the host, copy it to a desired location and set the environment variable:

  ```bash
  export RUNTIME_LIB_DIR=/your/copied/path/runtime_lib
  ```

When using `aiecc.py` to compile, we met various problems. 

If you encounter errors similar to 
```text
error: expected ')' at end of argument list
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
                                                          ^
``` 
To resolve this, you can patch the `downgrade_ir_for_peano` function in:
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