# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

file(GLOB globbed *.cpp)

add_mlir_library(MLIRAlloSupport
  ${globbed}

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSPIRVDialect
  MLIRTranslateLib
  MLIRAllo
  MLIRMemRefDialect
  MLIRAnalysis
  MLIRAffineAnalysis
)

if(BUILD_DATAFLOW)
  target_link_libraries(MLIRAlloSupport PUBLIC
    SdyDialect          # Linked from Shardy
    StablehloOps        # Linked from StableHLO
  )
endif()
