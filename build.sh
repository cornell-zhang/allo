#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

export ALLO_HOME=$(pwd)

# Pull the LLVM project and hcl-mlir dialect
git submodule update --init --recursive

# Check the Python version
if ! python3 -c 'import sys; assert sys.version_info >= (3,12)' > /dev/null; then
  echo "Error: Python 3.12 or later is required"
  exit 1
fi

# Install the required Python packages
echo "Installing the required Python packages ..."
python3 -m pip install -r requirements.txt

# Note: we need to patch the LLVM project to add additional
# supports for Python binding
echo "Patching LLVM project ..."
cp externals/llvm_patch externals/hcl_mlir/externals/llvm-project
cd externals/hcl_mlir/externals/llvm-project
git apply llvm_patch

# Install LLVM v18.x
# Make sure you are in the correct Python environment
echo "Installing LLVM v18.x ..."
mkdir -p build && cd build
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
export PATH=$LLVM_BUILD_DIR/bin:$PATH
echo "LLVM build directory: $LLVM_BUILD_DIR"

# Build the hcl dialect
echo "Building hcl dialect ..."
cd ../../..
mkdir -p build && cd build
cmake -G "Unix Makefiles" .. \
   -DMLIR_DIR=$LLVM_BUILD_DIR/lib/cmake/mlir \
   -DLLVM_EXTERNAL_LIT=$LLVM_BUILD_DIR/bin/llvm-lit \
   -DPYTHON_BINDING=ON \
   -DOPENSCOP=OFF \
   -DPython3_EXECUTABLE=`which python3` \
   -DCMAKE_CXX_FLAGS="-Wfatal-errors -std=c++17"
make -j8

# Install hcl dialect
echo "Installing hcl dialect ..."
cd tools/hcl/python_packages/hcl_core
python3 -m pip install -e .

# Install allo
echo "Installing allo ..."
cd $ALLO_HOME
python3 -m pip install -e .
echo "Installation completed!"
