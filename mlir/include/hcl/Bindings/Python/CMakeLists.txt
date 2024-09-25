# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

include(AddMLIRPython)

# The directory at which the Python import tree begins.
# See documentation for `declare_mlir_python_sources`'s ROOT_DIR
# argument.
set(HCL_MLIR_PYTHON_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/hcl")
set(HCL_MLIR_PYTHON_PACKAGES_DIR "${PROJECT_BINARY_DIR}/tools/hcl/python_packages")
set(MLIR_PYTHON_SOURCE_DIR "${MLIR_MAIN_SRC_DIR}/lib/Bindings/Python")
set(HCL_PYTHON_SOURCE_DIR "${PROJECT_SOURCE_DIR}/lib/Bindings/Python")

include_directories(${MLIR_PYTHON_SOURCE_DIR})

# We vendor our own MLIR instance in the `hcl_mlir` namespace.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=hcl_mlir.")

################################################################################
# Sources
################################################################################

declare_mlir_python_sources(HCLMLIRPythonSources)
declare_mlir_python_sources(HCLMLIRPythonExtensions)

declare_mlir_python_sources(HCLMLIRPythonSources.Dialects
  ROOT_DIR "${HCL_MLIR_PYTHON_ROOT_DIR}"
  ADD_TO_PARENT HCLMLIRPythonSources
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT HCLMLIRPythonSources.Dialects
  ROOT_DIR "${HCL_MLIR_PYTHON_ROOT_DIR}"
  TD_FILE dialects/HeteroCLBinding.td
  SOURCES
    dialects/hcl.py
    build_ir.py
    exceptions.py
    __init__.py
  DIALECT_NAME hcl
)

# Only a temporal solution for Affine dialect Python binding
# Ref: https://github.com/llvm/llvm-project/commit/8c1b785ce110b754c2112906021a929ddd32f587
declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT HCLMLIRPythonSources.Dialects
  ROOT_DIR "${HCL_MLIR_PYTHON_ROOT_DIR}"
  #TD_FILE dialects/AffineOps.td
  SOURCES
    dialects/affine.py
    dialects/_affine_ops_ext.py
    dialects/_affine_ops_gen.py
  DIALECT_NAME affine
)

################################################################################
# Extensions
################################################################################

declare_mlir_python_extension(HCLMLIRPythonExtensions.Main
  MODULE_NAME _hcl
  ADD_TO_PARENT HCLMLIRPythonExtensions
  ROOT_DIR "/"
  SOURCES
    ${HCL_PYTHON_SOURCE_DIR}/HCLModule.cpp
    ${HCL_PYTHON_SOURCE_DIR}/HCLTypes.cpp
    ${HCL_PYTHON_SOURCE_DIR}/HCLAttributes.cpp
  EMBED_CAPI_LINK_LIBS
    MLIRCAPIIR
    MLIRCAPIDebug
    MLIRHCLCAPI
    MLIRHCLCAPIEmitHLSCpp
  PRIVATE_LINK_LIBS
    MLIRPass
    MLIRHCLPasses
    LLVMSupport
)

################################################################################
# Generate packages and shared library
# Downstreams typically will not use these, but they are useful for local
# testing.
################################################################################

set(_source_components
  HCLMLIRPythonSources
  HCLMLIRPythonExtensions
  MLIRPythonSources
  MLIRPythonExtension.RegisterEverything
)

add_mlir_python_common_capi_library(HCLMLIRAggregateCAPI
  INSTALL_COMPONENT HCLMLIRPythonModules
  INSTALL_DESTINATION python_packages/hcl_core/hcl_mlir/_mlir_libs
  OUTPUT_DIRECTORY "${HCL_MLIR_PYTHON_PACKAGES_DIR}/hcl_core/hcl_mlir/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_HEADERS
    MLIRPythonCAPI.HeaderSources
  DECLARED_SOURCES
    ${_source_components}
)

add_mlir_python_modules(HCLMLIRPythonModules
  ROOT_PREFIX "${HCL_MLIR_PYTHON_PACKAGES_DIR}/hcl_core/hcl_mlir"
  INSTALL_PREFIX "python_packages/hcl_core/hcl_mlir"
  DECLARED_SOURCES ${_source_components}
  COMMON_CAPI_LINK_LIBS
    HCLMLIRAggregateCAPI
  )

add_custom_command(
        TARGET HCLMLIRPythonModules POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        ${HCL_MLIR_PYTHON_ROOT_DIR}/dialects/_linalg_ops_ext.py
        ${HCL_MLIR_PYTHON_PACKAGES_DIR}/hcl_core/hcl_mlir/dialects)

################################################################################
# Add setup script
################################################################################

add_custom_command(
          TARGET HCLMLIRPythonModules POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy
          ${HCL_MLIR_PYTHON_ROOT_DIR}/../setup.py
          ${HCL_MLIR_PYTHON_PACKAGES_DIR}/hcl_core)
