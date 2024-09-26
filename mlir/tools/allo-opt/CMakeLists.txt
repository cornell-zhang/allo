# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

set(LLVM_LINK_COMPONENTS
  Core
  Support
  nativecodegen
  OrcJIT
  )
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIRExecutionEngine
        MLIRLLVMCommonConversion
        MLIRLLVMDialect
        MLIRLLVMToLLVMIRTranslation
        MLIRTargetLLVMIRExport
        MLIRIR
        MLIROptLib
        MLIRAllo
        MLIRAlloConversion
        MLIRAlloPasses
        )
add_llvm_executable(allo-opt allo-opt.cpp)

llvm_update_compile_flags(allo-opt)
target_link_libraries(allo-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(allo-opt)
