# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

set(LLVM_TARGET_DEFINITIONS HeteroCLOps.td)
mlir_tablegen(HeteroCLOps.h.inc -gen-op-decls)
mlir_tablegen(HeteroCLOps.cpp.inc -gen-op-defs)
mlir_tablegen(HeteroCLDialect.h.inc -gen-dialect-decls -dialect=hcl)
mlir_tablegen(HeteroCLDialect.cpp.inc -gen-dialect-defs -dialect=hcl)
add_public_tablegen_target(MLIRHeteroCLOpsIncGen)
add_dependencies(mlir-headers MLIRHeteroCLOpsIncGen)

set(LLVM_TARGET_DEFINITIONS HeteroCLTypes.td)
mlir_tablegen(HeteroCLTypes.h.inc -gen-typedef-decls)
mlir_tablegen(HeteroCLTypes.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(MLIRHeteroCLTypesIncGen)

set(LLVM_TARGET_DEFINITIONS HeteroCLAttrs.td)
mlir_tablegen(HeteroCLAttrs.h.inc -gen-attrdef-decls)
mlir_tablegen(HeteroCLAttrs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRHeteroCLAttrsIncGen)

set(LLVM_TARGET_DEFINITIONS HeteroCLAttrs.td)
mlir_tablegen(HeteroCLEnums.h.inc -gen-enum-decls)
mlir_tablegen(HeteroCLEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRHeteroCLEnumsIncGen)

add_mlir_doc(HeteroCLDialect HeteroCLDialect HeteroCL/ -gen-dialect-doc)
add_mlir_doc(HeteroCLOps HeteroCLOps HeteroCL/ -gen-op-doc)
