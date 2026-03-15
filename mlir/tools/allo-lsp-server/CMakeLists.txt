# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_tool(allo-lsp-server allo-lsp-server.cpp)

llvm_update_compile_flags(allo-lsp-server)

add_dependencies(allo-lsp-server MLIRAlloIncGen)

target_link_libraries(allo-lsp-server
    PRIVATE
    ${dialect_libs}
    ${conversion_libs}
    MLIRIR
    MLIRLspServerLib
    MLIRAllo
    MLIRAlloTransformOps
)

