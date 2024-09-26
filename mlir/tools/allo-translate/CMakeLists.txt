# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

set(LLVM_LINK_COMPONENTS
  Support
  )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

add_llvm_executable(allo-translate
  allo-translate.cpp
  )
llvm_update_compile_flags(allo-translate)

if(OPENSCOP)
    target_link_libraries(allo-translate
      PRIVATE
      ${dialect_libs}
      ${translation_libs}
      MLIRIR
      MLIRParser
      MLIRPass
      MLIRSPIRVDialect
      MLIRTranslateLib
      MLIRSupport
      MLIRAlloEmitHLSCpp
      MLIRAlloEmitOpenSCoP
      MLIRAllo
      gmp
      )
else()
    target_link_libraries(allo-translate
      PRIVATE
      ${dialect_libs}
      ${translation_libs}
      MLIRIR
      MLIRParser
      MLIRPass
      MLIRSPIRVDialect
      MLIRTranslateLib
      MLIRSupport
      MLIRAlloEmitHLSCpp
      MLIRAllo
      )
endif()


mlir_check_link_libraries(allo-translate)
