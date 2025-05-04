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