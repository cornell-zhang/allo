<!--- Copyright Allo authors. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# Allo DSL

Please refer to [Documentation](https://chhzh123.github.io/allo-docs) for more details.


## Installation

Please clone the Allo repository to your local machine.

```bash
git clone https://github.com/cornell-zhang/allo.git
cd allo
```

We recommend creating a new conda environment for Allo. The default Python version is 3.8.

```bash
conda create -n allo python=3.8
conda activate allo
```

### Prerequisites

We need to first install the [LLVM project](https://github.com/llvm/llvm-project/tree/llvmorg-18-init) and the [hcl-mlir dialect](https://github.com/cornell-zhang/hcl-dialect). Users can choose to build from source or use our provided docker.

#### Build from source

```bash
# Pull the LLVM project and hcl-mlir dialect
git submodule update --init --recursive

# Note: we need to patch the LLVM project to add additional
# supports for Python binding
cp externals/llvm_patch externals/hcl_mlir/externals/llvm-project
cd externals/hcl_mlir/externals/llvm-project
git apply llvm_patch

# Install LLVM v18.x
# Make sure you are in the correct Python environment
mkdir build && cd build
cmake -G "Unix Makefiles" ../llvm \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="host" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DLLVM_INSTALL_UTILS=ON \
   -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
   -DPython3_EXECUTABLE=`which python3`
make -j8

# Export the LLVM build directory
export LLVM_BUILD_DIR=$(pwd)

# Build the hcl dialect
cd ../../..
mkdir build && cd build
cmake -G "Unix Makefiles" .. \
   -DMLIR_DIR=$LLVM_BUILD_DIR/lib/cmake/mlir \
   -DLLVM_EXTERNAL_LIT=$LLVM_BUILD_DIR/bin/llvm-lit \
   -DPYTHON_BINDING=ON \
   -DOPENSCOP=OFF \
   -DPython3_EXECUTABLE=`which python3` \
   -DCMAKE_CXX_FLAGS="-Wfatal-errors -std=c++17"
make -j8

# Install hcl dialect
cd tools/hcl/python_packages/hcl_core
python3 -m pip install -e .
```

#### Docker

To simplify the installation process, we provide a docker image that has already installed the LLVM-18.x project.
Please pull the image from Docker Hub, **patch LLVM, and install the hcl dialect** as described above.

```bash
# The llvm is installed in /root/llvm-project in the docker image
docker pull chhzh123/llvm-project:18.x
```


### Install Allo

After installing LLVM and the hcl dialect, we can directly `pip install` Allo:

```bash
# Under the root directory of Allo
python3 -m pip install -e .
```
