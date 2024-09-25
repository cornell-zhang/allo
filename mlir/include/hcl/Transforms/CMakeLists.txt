# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls)
add_public_tablegen_target(MLIRHeteroCLPassesIncGen)

add_mlir_doc(HeteroCLPass HeteroCLPass HeteroCL/ -gen-pass-doc)