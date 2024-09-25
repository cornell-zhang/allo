# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

set(LLVM_TARGET_DEFINITIONS HCLTransformOps.td)
mlir_tablegen(HCLTransformOps.h.inc -gen-op-decls)
mlir_tablegen(HCLTransformOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRHCLTransformOpsIncGen)
