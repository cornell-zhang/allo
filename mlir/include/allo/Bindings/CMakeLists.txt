# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

include(AddMLIRPython)

# The directory at which the Python import tree begins.
# See documentation for `declare_mlir_python_sources`'s ROOT_DIR
# argument.
set(ALLO_MLIR_PYTHON_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/allo")
set(ALLO_MLIR_PYTHON_PACKAGES_DIR "${PROJECT_BINARY_DIR}/tools/allo")
set(MLIR_PYTHON_SOURCE_DIR "${MLIR_MAIN_SRC_DIR}/lib/Bindings")
set(ALLO_PYTHON_SOURCE_DIR "${PROJECT_SOURCE_DIR}/lib/Bindings")

include_directories(${MLIR_PYTHON_SOURCE_DIR})

# We vendor our own MLIR instance in the `allo._mlir` namespace.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=allo._mlir.")

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

file(GLOB_RECURSE MLIR_PYTHON_EXTRA_SOURCES
  RELATIVE "${PROJECT_SOURCE_DIR}/../externals/mlir-python-extra/mlir"
  "${PROJECT_SOURCE_DIR}/../externals/mlir-python-extra/mlir/extras/*.py"
)
declare_mlir_python_sources(MlirPythonExtraSources
  ROOT_DIR "${PROJECT_SOURCE_DIR}/../externals/mlir-python-extra/mlir"
  SOURCES
    ${MLIR_PYTHON_EXTRA_SOURCES}
)
list(APPEND _source_components MlirPythonExtraSources)


add_mlir_python_common_capi_library(AlloMLIRAggregateCAPI
  INSTALL_COMPONENT AlloMLIRPythonModules
  INSTALL_DESTINATION _mlir
  OUTPUT_DIRECTORY "${ALLO_MLIR_PYTHON_PACKAGES_DIR}/_mlir"
  RELATIVE_INSTALL_ROOT "../.."
  DECLARED_HEADERS
    MLIRPythonCAPI.HeaderSources
  DECLARED_SOURCES
    ${_source_components}
)

add_mlir_python_modules(AlloMLIRPythonModules
  ROOT_PREFIX "${ALLO_MLIR_PYTHON_PACKAGES_DIR}/_mlir"
  INSTALL_PREFIX "_mlir"
  DECLARED_SOURCES ${_source_components}
  COMMON_CAPI_LINK_LIBS
    AlloMLIRAggregateCAPI
  )
