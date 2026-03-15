# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

set(LLVM_TARGET_DEFINITIONS AlloTransformOps.td)
mlir_tablegen(AlloTransformOps.h.inc -gen-op-decls)
mlir_tablegen(AlloTransformOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(AlloTransformOpsIncGen)

add_dependencies(MLIRAlloIncGen AlloTransformOpsIncGen)