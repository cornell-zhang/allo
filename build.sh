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

# Install LLVM v19.x
# Make sure you are in the correct Python environment
echo "Installing LLVM v19.x ..."
mkdir -p build && cd build
cmake -G Ninja ../llvm \
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

# Install allo
echo "Installing allo ..."
cd $ALLO_HOME
python3 -m pip install -e .
echo "Installation completed!"
