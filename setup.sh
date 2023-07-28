#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# LLVM
export LLVM_HOME=/work/shared/common/llvm-project-hcl
export PREFIX=$LLVM_HOME
export LLVM_BUILD_DIR=$LLVM_HOME/build
export LLVM_SYMBOLIZER_PATH=${LLVM_BUILD_DIR}/bin/llvm-symbolizer
export PATH=$LLVM_BUILD_DIR/bin:$PATH
# LLVM Python Bindings
export PYTHONPATH=$LLVM_HOME/build/tools/mlir/python_packages/mlir_core:$PYTHONPATH

# HCL dialect
export HCL_DIALECT_BUILD_DIR=/work/shared/common/hcl-dialect/build
export PYTHONPATH=${HCL_DIALECT_BUILD_DIR}/tools/hcl/python_packages/hcl_core:${PYTHONPATH}
export PATH=${HCL_DIALECT_BUILD_DIR}/bin:$PATH

# HCL Runtime Library
export LD_LIBRARY_PATH=${HCL_DIALECT_BUILD_DIR}/lib:$LD_LIBRARY_PATH

# Allo package
# Please intall by yourself
