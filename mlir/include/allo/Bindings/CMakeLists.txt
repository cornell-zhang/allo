# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

include(AddMLIRPython)

# The directory at which the Python import tree begins.
# See documentation for `declare_mlir_python_sources`'s ROOT_DIR
# argument.
set(ALLO_MLIR_PYTHON_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/allo")
set(ALLO_MLIR_PYTHON_PACKAGES_DIR "${PROJECT_BINARY_DIR}/tools/allo/python_packages")
set(MLIR_PYTHON_SOURCE_DIR "${MLIR_MAIN_SRC_DIR}/lib/Bindings/Python")
set(ALLO_PYTHON_SOURCE_DIR "${PROJECT_SOURCE_DIR}/lib/Bindings/Python")

include_directories(${MLIR_PYTHON_SOURCE_DIR})

# We vendor our own MLIR instance in the `allo_mlir` namespace.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=allo_mlir.")

################################################################################
# Sources
################################################################################

declare_mlir_python_sources(AlloMLIRPythonSources)
declare_mlir_python_sources(AlloMLIRPythonExtensions)

declare_mlir_python_sources(AlloMLIRPythonSources.Dialects
  ROOT_DIR "${ALLO_MLIR_PYTHON_ROOT_DIR}"
  ADD_TO_PARENT AlloMLIRPythonSources
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT AlloMLIRPythonSources.Dialects
  ROOT_DIR "${ALLO_MLIR_PYTHON_ROOT_DIR}"
  TD_FILE dialects/AlloBinding.td
  SOURCES
    dialects/allo.py
    exceptions.py
    __init__.py
  DIALECT_NAME allo
)

# Only a temporal solution for Affine dialect Python binding
# Ref: https://github.com/llvm/llvm-project/commit/8c1b785ce110b754c2112906021a929ddd32f587
# declare_mlir_dialect_python_bindings(
#   ADD_TO_PARENT AlloMLIRPythonSources.Dialects
#   ROOT_DIR "${ALLO_MLIR_PYTHON_ROOT_DIR}"
#   #TD_FILE dialects/AffineOps.td
#   SOURCES
#     dialects/affine.py
#     dialects/_affine_ops_ext.py
#     dialects/_affine_ops_gen.py
#   DIALECT_NAME affine
# )

################################################################################
# Extensions
################################################################################

declare_mlir_python_extension(AlloMLIRPythonExtensions.Main
  MODULE_NAME _allo
  ADD_TO_PARENT AlloMLIRPythonExtensions
  ROOT_DIR "/"
  SOURCES
    ${ALLO_PYTHON_SOURCE_DIR}/AlloModule.cpp
    ${ALLO_PYTHON_SOURCE_DIR}/AlloTypes.cpp
    ${ALLO_PYTHON_SOURCE_DIR}/AlloAttributes.cpp
  EMBED_CAPI_LINK_LIBS
    MLIRCAPIIR
    MLIRCAPIDebug
    MLIRAlloCAPI
    MLIRAlloCAPIEmitHLSCpp
  PRIVATE_LINK_LIBS
    MLIRPass
    MLIRAlloPasses
    LLVMSupport
)

################################################################################
# Generate packages and shared library
# Downstreams typically will not use these, but they are useful for local
# testing.
################################################################################

set(_source_components
  AlloMLIRPythonSources
  AlloMLIRPythonExtensions
  MLIRPythonSources
  MLIRPythonExtension.RegisterEverything
)

add_mlir_python_common_capi_library(AlloMLIRAggregateCAPI
  INSTALL_COMPONENT AlloMLIRPythonModules
  INSTALL_DESTINATION python_packages/allo_core/allo_mlir/_mlir_libs
  OUTPUT_DIRECTORY "${ALLO_MLIR_PYTHON_PACKAGES_DIR}/allo_core/allo_mlir/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_HEADERS
    MLIRPythonCAPI.HeaderSources
  DECLARED_SOURCES
    ${_source_components}
)

add_mlir_python_modules(AlloMLIRPythonModules
  ROOT_PREFIX "${ALLO_MLIR_PYTHON_PACKAGES_DIR}/allo_core/allo_mlir"
  INSTALL_PREFIX "python_packages/allo_core/allo_mlir"
  DECLARED_SOURCES ${_source_components}
  COMMON_CAPI_LINK_LIBS
    AlloMLIRAggregateCAPI
  )

################################################################################
# Add setup script
################################################################################

add_custom_command(
          TARGET AlloMLIRPythonModules POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy
          ${ALLO_MLIR_PYTHON_ROOT_DIR}/setup.py
          ${ALLO_MLIR_PYTHON_PACKAGES_DIR}/allo_core)
