# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

add_mlir_dialect_library(MLIRHCLTransformOps
  HCLTransformOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/hcl/Dialect/TransformOps

  DEPENDS
  MLIRHCLTransformOpsIncGen

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRFuncDialect
  MLIRIR
  MLIRPDLDialect
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRSCFUtils
  MLIRTransformDialect
  MLIRVectorDialect
)
