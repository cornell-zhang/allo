# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

file(GLOB globbed *.cpp)

add_mlir_library(MLIRAlloEmitHLSCpp
  ${globbed}

  LINK_LIBS PUBLIC
  MLIRAllo
  MLIRAlloSupport
  MLIRMemRefDialect
  MLIRAnalysis
)