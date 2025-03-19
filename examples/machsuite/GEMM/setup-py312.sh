#!/bin/bash

# Enable gcc-7
source /opt/rh/devtoolset-7/enable

# LLVM
export LLVM_HOME=/work/shared/users/common/llvm-project-18.x
export PREFIX=$LLVM_HOME
export LLVM_BUILD_DIR=$LLVM_HOME/build-patch
export LLVM_SYMBOLIZER_PATH=${LLVM_BUILD_DIR}/bin/llvm-symbolizer
export PATH=$LLVM_BUILD_DIR/bin:$PATH
# LLVM Python Bindings
export PYTHONPATH=$LLVM_HOME/build/tools/mlir/python_packages/mlir_core:$PYTHONPATH

# HCL dialect
export HCL_DIALECT_BUILD_DIR=/work/shared/users/common/hcl-dialect-18.x/build-py312
export PYTHONPATH=${HCL_DIALECT_BUILD_DIR}/tools/hcl/python_packages/hcl_core:${PYTHONPATH}
export PATH=${HCL_DIALECT_BUILD_DIR}/bin:$PATH

# HCL Runtime Library
export LD_LIBRARY_PATH=${HCL_DIALECT_BUILD_DIR}/lib:$LD_LIBRARY_PATH

# Allo package
# Please intall by yourself