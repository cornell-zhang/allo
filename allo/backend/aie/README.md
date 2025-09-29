<!--- Copyright Allo authors. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# Experimental MLIR-AIE Codegen
- [Environment Setup](#environment-setup)
- [Usage](#usage)
    - [Basic Examples](#example)
    - [Environment Variables for Controlling Compiler Behavior](#environment-variables-for-controlling-compiler-behavior)
    - [New Features](#new-feature)
        - [Timing-based Profiling](#profiling)
        - [Tracing-based Profiling](#profiling-with-trace)
        - [Customized External Kernel](#support-for-user-defined-external-kernels)

## Environment Setup
Please follow the [Getting Started](https://github.com/Xilinx/mlir-aie/tree/main?tab=readme-ov-file#getting-started-for-amd-ryzen-ai-on-linux) guide to install MLIR-AIE.

In **Step 3: Install IRON library, mlir-aie, and llvm-aie compilers from wheels**, under the section [Install IRON for AMD Ryzen™ AI AIE Application Development](https://github.com/Xilinx/mlir-aie/tree/main?tab=readme-ov-file#install-iron-for-amd-ryzen-ai-aie-application-development), please install version `v1.0` using the following commands:
```bash
# Install IRON library and mlir-aie from a wheel
python3 -m pip install mlir_aie -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/v1.0

# Install Peano from a llvm-aie wheel
python3 -m pip install https://github.com/Xilinx/llvm-aie/releases/download/nightly/llvm_aie-19.0.0.2025041501+b2a279c1-py3-none-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
```
> ⚠️ **Note:** The mlir_aie wheel require `manylinux_2_35`, and some systems (e.g., those with glibc 2.34, can be confirmed by `ldd --version`) do not meet this requirement. This results in an installation failure like:
> `ERROR: mlir_aie-0.0.1.2025042204+24208c0-cp312-cp312-manylinux_2_35_x86_64.whl is not a supported wheel on this platform.`

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

To enable the experimental MLIR-AIE codegen, specify `"aie"` as the target in the `dataflow.build` function.

Currently, the supported target platforms include `XDNA1` and `XDNA2`.
By default, the target platform is set to `XDNA1`.
To switch to `XDNA2`, please run:
```bash
export NPU2=1  
```

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
        mod = df.build(top, target="aie")
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

    mod = df.build(top, target="aie")
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
        mod = df.build(top, target="aie")
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


mod1 = df.build(top1, target="aie", project="top1.prj")
mod2 = df.build(top2, target="aie", project="top2.prj")

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

### Environment Variables for Controlling Compiler Behavior
The compiler exposes several environment variables that can be used to fine-tune its behavior.

You can set them globally in the terminal, for example:
```bash
export FORCE_UNROLL_INDEX=1
```
Or only for a specific test case in Python:

```python
import os

os.environ["FORCE_UNROLL_INDEX"] = "1"
# run your test
del os.environ["FORCE_UNROLL_INDEX"]  # reset after use
```

#### `FORCE_UNROLL_INDEX`

By default, the AIE backend tries to **avoid unrolling** `meta_for` loops to optimize code size.
However, when the iterator of a rolled `meta_for` is used as the index of a pipe, the current virtual mapping (especially chain) faces significant restrictions.

If you prefer to sacrifice code size in exchange for greater flexibility in using mapping primitives, you can set:

```
FORCE_UNROLL_INDEX=1
```

This will force the compiler to **unroll any `meta_for` loop whose iterator is used as an index**, bypassing the default optimization.

#### `ENABLE_AGGRESSIVE_PORT_UTILIZATION_PATCH`
This variable controls an optimization patch for DMA scheduling.

If you want to maximize DMA performance, you can try:
```
ENABLE_AGGRESSIVE_PORT_UTILIZATION_PATCH=1
```
This enables an aggressive strategy that tries to fully utilize available DMA ports.

#### `DEBUG`

This variable controls whether virtual mapping graphs are dumped.

You can use the virtual mapping graph to identify the available virtual mapping primitives, understand how to apply them and verify if the mapping behaves as expected. 

If you want to dump the virtual mapping graphs for debugging, you can set:
```
DEBUG=1
```
After building the dataflow module, you can find the visualization result as a PDF file in the build project directory.

##### Example 
The visualization result for [`tests/dataflow/aie/test_cannon.py`](../../../tests/dataflow/aie/test_cannon.py) can be found as a PDF file named "virtual_graph.pdf" in the project directory (`top.prj`).

<img width="80%" alt="cannon" src="https://github.com/user-attachments/assets/37b6db68-92f5-4961-b41e-b4af1fb7fb2d" />

For programs that use virtual mapping primitives, two visualization results are generated (using `_test_split_k_gemm_2x2x4` in `tests/dataflow/aie/test_mapping_gemm.py` as an example):

- The original virtual mapping graph (named "virtual_graph.pdf")
    <img width="60%" alt="before" src="https://github.com/user-attachments/assets/884e183f-1c36-4e89-83dc-99f9d415861e" />


- The virtual mapping graph after applying all mapping primitives (named "after_mapping.pdf")
    <img width="60%" alt="after" src="https://github.com/user-attachments/assets/22f82618-68a1-409e-b8c7-d10d252ef85a" />

#### 
### New Feature
#### Profiling
A new timing-based profiling feature has been added to help measure the performance of the module during execution. 

To enable profiling, use the `profile` flag in the `build` method in [`dataflow.py`](../../dataflow.py):
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
    mapping_primitives: list[tuple[str, list]] = [],
    profile=False,
    warmup=20,
    num_iters=100,
    trace: list[tuple[str, tuple[int, ...]]] = None,
    trace_size: int = 4096,
    device_type: str = None,
)
```

**Related Parameters:**

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
    target="aie",
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
#### Profiling with Trace
AIEs are equipped with tracing hardware that provides a cycle accurate view of hardware events. 
This enables more precise profiling, especially for analyzing the performance of computation on each compute tile (AIE) and the associated data transfers.

However, configuring the trace unit can be complex. This new feature simplifies the process, making trace-based profiling easier to use.

Trace-based profiling requires configuring the compute tile and routing the trace data as packets through the shim tile to external memory. 
This places additional pressure on the DMA ports of the shim tile, making it unsuitable for large-scale computation tasks where DMA bandwidth is already a constrained resource. 
As a result, trace support is currently provided mainly for small-scale computations.

To use trace, users can configure the options in the `build` method in [`dataflow.py`](../../dataflow.py):
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
    mapping_primitives: list[tuple[str, list]] = [],
    profile=False,
    warmup=20,
    num_iters=100,
    trace: list[tuple[str, tuple[int, ...]]] = None,
    trace_size: int = 4096,
    device_type: str = None,
):
```

**Related Parameters:**
- `trace` is a list of tiles from the `allo.dataflow.kernel` users wishes to trace. Each element consists of the kernel’s name as a string and a tuple representing the tile index. Note that this index does not necessarily correspond to the final physical compute tile index in the 2D AIE array. Also note that due to resource constraints, tracing is enabled on a best-effort basis. If resources such as DMA ports or buffer descriptors are limited, tracing may not be applied to all specified tiles in the `trace` list.
- `trace_size` specifies the size of the trace buffer. If a large amount of trace information is expected, users may increase trace_size accordingly.

After `build`, running the generated module will produce a file named `trace.txt` under the `project` directory. 

The `trace.txt` file should contain multiple lines of non-zero values.
If all entries are zero, please first check whether the `top.mlir` file contains any `aie.packet_flow` operations.
- If not, it indicates that tracing for the specified tiles was skipped due to resource constraints.
- If such operations are present but entries in `trace.txt` are all zero, please submit a bug report.

Users can use multiple tool to parse the `trace.txt` and convert it into a more human-readable format. 
Some of the useful parsers can be found in the `mlir-aie` repository. For example, [`parse_trace.py`](https://github.com/Xilinx/mlir-aie/blob/v1.0/programming_examples/utils/parse_trace.py) parses it into a json file and users can use [Perfetto](http://ui.perfetto.dev) to view the timeline (check [this link](https://github.com/Xilinx/mlir-aie/blob/v1.0/programming_examples/utils/README.md#trace-parser-parse_tracepy) for more details).

> **Note:** the unit of timing reported in perfetto should be interpreted as cycle count (check [this issue](https://github.com/Xilinx/mlir-aie/issues/2214) for more information).

##### Example
Tracing tile `(0, 0)` of the `allo.dataflow.kernel` named `gemm`.
```python
TyI, TyO = int16, int32
M, N, K = 32, 32, 32
P0, P1 = 2, 4

@df.region()
def top():
    @df.kernel(mapping=[P0, P1])
    def gemm(A: TyI[M, K] @ LyA, B: TyI[K, N] @ LyB, C: TyO[M, N] @ LyC):
        C[:, :] = allo.matmul(A, B)

# trace tile (0, 0) of gemm df.kernel
mod = df.build(
    top,
    target="aie",
    trace=[
        ("gemm", (0, 0)),
    ],
    trace_size=65536,
)
A = np.random.randint(0, 64, (M, K)).astype(np.int16)
B = np.random.randint(0, 64, (K, N)).astype(np.int16)
C = np.zeros((M, N)).astype(np.int32)
mod(A, B, C)
np_C = A.astype(np.int32) @ B.astype(np.int32)
np.testing.assert_allclose(C, np_C, atol=1e-5)
print("PASSED!")
```
##### Using Trace to Measure the Performance of External Kernels
Trace is useful for evaluating the performance of an external kernel running on a single compute tile. 
This is especially important when profiling for optimizations such as vectorization of external kernels. The following example demonstrates how to use trace profiling on some [convolution kernels](../../library/aie/).

In this case, due to the relatively small computation scale, the difference between the [vectorized](../../library/aie/conv_small_vector.cc) and [scalar](../../library/aie/conv_small_scalar.cc) versions of the kernel is not clearly observable using timing-based profiling to measure NPU time. 
Instead, one can insert event markers, such as `event0();` and `event1();`, directly into the external C++ code and run the trace on the compute tile executing the external kernel. Sample code can be found in [`test_trace_conv.py`](../../../tests/dataflow/aie/test_trace_conv.py).

Process the generated trace (in `top.prj/trace.txt`) with [`parse_trace.py`](https://github.com/Xilinx/mlir-aie/blob/v1.0/programming_examples/utils/parse_trace.py).
```bash
# sample processing cmds
cd top.prj
path/to/parse_trace.py --filename trace.txt --mlir top.mlir --colshift 1 > trace_scalar.json
```
And use [Perfetto](http://ui.perfetto.dev) to view the timeline.
The timeline view reveals a clear performance difference between the two external kernel versions.

- scalar version
    
    <img width="80%" alt="scalar" src="https://github.com/user-attachments/assets/4cc92e2b-4b4c-495d-8718-0c5d32d22c00" />
  
- vector version

    <img width="80%" alt="vector" src="https://github.com/user-attachments/assets/4c5b558d-c84d-4c16-aef2-3c626b62bbee" />

From the timeline screenshot, you can observe a clear difference in the computation cycle count between the two kernels within the regions marked by the event markers. 
Additionally, you can see that the vectorized version makes use of vector instructions, which are absent in the scalar version.

If you need more precise cycle counts or additional profiling information, you can write your own processing script to analyze the generated JSON file, or directly parse the `trace.txt`.

#### Support for user-defined external kernels

Originally, complex computations on AIE cores were implemented using a limited set of [external kernels provided in the `mlir-aie` repository](https://github.com/Xilinx/mlir-aie/tree/v1.0/aie_kernels). However, this external kernel library supports only a narrow range of operations and leaves room for performance improvement. To address these limitations, we add support for user-defined external kernels.

Users can now register and invoke external kernels implemented in C++ and exposed via extern "C" interfaces. These kernels can be written using the AIE API and integrated into the programming model workflow.

Suppose the external kernel is implemented in the `norm.cc` file:
```cpp
#include <aie_api/aie.hpp>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#define NOCPP

#define EPS 1e-6f // epsilon

template <typename T_in, typename T_out, const int SEQ_LEN, const int HIDDEN>
void rms_norm_single_batch(T_in *input_tensor, T_in *weight,
                           T_out *output_tensor) {
  constexpr int vec_factor = 16;
  using vec_t = aie::vector<T_in, vec_factor>;
  event0();
  for (int iter = 0; iter < SEQ_LEN; iter++) {
    T_in *__restrict input_ptr = input_tensor;
    T_in *__restrict weight_ptr = weight;
    T_out *__restrict output_ptr = output_tensor;
    float square_sum = 0.0f;
    const int F = HIDDEN / vec_factor;
    for (int i = 0; i < F; i++) {
      vec_t input_vec = aie::load_v<vec_factor>(input_ptr);
      input_ptr += vec_factor;
      vec_t square_vec = aie::mul(input_vec, input_vec);
      square_sum += aie::reduce_add(square_vec);
    }
    vec_t square_sum_vec =
        aie::broadcast<T_in, vec_factor>(square_sum / HIDDEN + EPS);
    vec_t rms = aie::invsqrt(square_sum_vec);
    input_ptr = input_tensor;
    for (int i = 0; i < F; i++) {
      vec_t input_vec = aie::load_v<vec_factor>(input_ptr);
      input_ptr += vec_factor;
      vec_t normed = aie::mul(input_vec, rms);
      vec_t weight_vec = aie::load_v<vec_factor>(weight_ptr);
      weight_ptr += vec_factor;
      vec_t result = aie::mul(normed, weight_vec);
      aie::store_v(output_ptr, result);
      output_ptr += vec_factor;
    }
    input_tensor += HIDDEN;
    output_tensor += HIDDEN;
  }
  event1();
}
```
and exposed via extern "C" interfaces
```cpp
extern "C" {
  void layer_norm(float A_in[4][512], float B_in[512], float C_out[4][512]) {
    rms_norm_single_batch<float, float, 4, 512>(&A_in[0][0], B_in, &C_out[0][0]);
  }
}
```
> ⚠️ **Note:** External kernel function arguments must have fully specified constant shapes. Pointer types are not allowed.

Users can create an [ExternalModule](external_kernel.py) to wrap the kernel and use it in computation on AIE core.

Register the `ExternalModule` in the context.
```python
norm = ExternalModule(
    top="layer_norm",       # Name of the top-level function defined with `extern "C"`
    impl_path="norm.cc",    # Path to the user-provided source file that implements the external kernel
    input_idx=[0, 1],       # Indices of input arguments in the argument list passed to the module
    output_idx=[2],         # Indices of output arguments in the argument list passed to the module
)
```
And the external module can then be used in an Allo kernel.
```python
@df.kernel(mapping=[1])
    def core(A: Ty[M, N] @ LyA, B: Ty[N] @ Ly, C: Ty[M, N] @ LyA):
        norm(A, B, C)
```

An example can be found in [`tests/dataflow/aie/test_norm.py`](../../../tests/dataflow/aie/test_norm.py).

##### Allo External Kernel Library 
The [`kernels`](../../library/aie/) directory contains several external kernels used in the GPT-2 block.
Corresponding tests can be found in [`tests/dataflow/aie/gpt2`](../../../tests/dataflow/aie/gpt2/).
