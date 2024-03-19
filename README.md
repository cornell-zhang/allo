<!--- Copyright Allo authors. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# Allo: A Programming Model for Composable Accelerator Design

[**Documentation**](https://cornell-zhang.github.io/allo) 

![GitHub](https://img.shields.io/github/license/cornell-zhang/allo)
[![CircleCI](https://circleci.com/gh/cornell-zhang/allo.svg?style=shield)](https://circleci.com/gh/cornell-zhang/allo.svg?style=shield)

Allo is an Accelerator Design Language (ADL) and compiler that facilitates the construction of large-scale, high-performance hardware accelerators in a modular and composable manner. Allo has several key features:
* **Progressive hardware customizations**: Allo decouples hardware customizations from algorithm specifications and treats each hardware customization as a primitive that performs a rewrite on the program. Allo not only decouples the loop-based transformations, but also extends the decoupling to memory, communication, and data types.
* **Reusable parameterized kernel templates**: Allo supports declaring type variables during kernel creation and instantiating the kernel when building the hardware executable, which is an important feature for building reusable hardware kernel libraries. Allo introduces a concise grammar for creating kernel templates, eliminating the need for users to possess complicated metaprogramming expertise.
* **Composable schedules**: Allo empowers users to construct kernels incrementally from the bottom up, adding customizations one at a time while validating the correctness of each submodule. Ultimately, multiple schedules are progressively integrated into a complete design using the `.compose()` primitive. This approach, unachievable by prior top-down methods, significantly enhances productivity and debuggability.


## Installation

Please clone the Allo repository to your local machine.

```bash
git clone https://github.com/cornell-zhang/allo.git
cd allo
```

We recommend creating a new conda environment for Allo. Since we are using the latest Python features, the minimum Python version is **3.12**.

```bash
conda create -n allo python=3.12
conda activate allo
```


### Prerequisites

We need to first install the [LLVM project](https://github.com/llvm/llvm-project/tree/llvmorg-18-init) and the [hcl-mlir dialect](https://github.com/cornell-zhang/hcl-dialect). Users can choose to use our provided docker or build from source.

#### Docker

To simplify the installation process, we provide a docker image that has already installed the LLVM-18.x project.
Please pull the image from Docker Hub, **patch LLVM, and install the hcl dialect** as described above.

```bash
# * The LLVM is installed in /root/llvm-project in the docker image, which has already been patched
# * A prebuilt hcl-dialect is installed in /root/hcl-dialect, but please note that it is not up-to-date
#   You can pull the latest hcl-dialect using `git pull` and rebuild it if needed
docker pull chhzh123/hcl-dialect:llvm-18.x-py3.12
docker run --rm -it chhzh123/hcl-dialect:llvm-18.x-py3.12
```

#### Build from source

Users can also choose to build LLVM and the hcl dialect from source. Please follow the instructions below.

```bash
# Make sure you are under the correct Python environment
bash build.sh
```


### Install Allo

After installing LLVM and the hcl dialect, we can directly `pip install` Allo:

```bash
# Under the root directory of Allo
python3 -m pip install -e .
```

## Getting Started
Below is a minimal example of leveraging Allo to customize a GEMM kernel:
```python
import allo
from allo.ir.types import int32

# Allo kernel definition
def gemm(A: int32[32, 32], B: int32[32, 32]) -> int32[32, 32]:
    C: int32[32, 32] = 0
    for i, j, k in allo.grid(32, 32, 32):
        C[i, j] += A[i, k] * B[k, j]
    return C

# Schedule construction
s = allo.customize(gemm)

# Real-time transformation
s.split("i", factor=8)
print(s.module)

# Compilation
mod = s.build(target="llvm")

# Execution
import numpy as np
np_A = np.random.randint(0, 100, (32, 32)).astype(np.int32)
np_B = np.random.randint(0, 100, (32, 32)).astype(np.int32)
np_C = mod(np_A, np_B)

# Testing
golden_C = np.matmul(np_A, np_B)
np.testing.assert_allclose(np_C, golden_C, rtol=1e-5, atol=1e-5)
```

## Related Projects
* Accelerator Programming Languages: [Exo](https://github.com/exo-lang/exo), [Halide](https://github.com/halide/Halide), [TVM](https://github.com/apache/tvm)
* Accelerator Design Languages: [Dahlia](https://github.com/cucapra/dahlia), [HeteroCL](https://github.com/cornell-zhang/heterocl), [PyLog](https://github.com/hst10/pylog), [ScaleHLS](https://github.com/hanchenye/scalehls), [Spatial](https://github.com/stanford-ppl/spatial)
* Compiler Frameworks: [MLIR](https://mlir.llvm.org/)
