# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

file(GLOB globbed *.cpp)

add_mlir_library(MLIRHCLEmitHLSCpp
  ${globbed}

  LINK_LIBS PUBLIC
  MLIRHeteroCL
  MLIRHCLSupport
  MLIRMemRefDialect
  MLIRAnalysis
)