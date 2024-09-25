# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

file(GLOB globbed *.cpp)

get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_mlir_dialect_library(MLIRHCLConversion
    ${globbed}

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/hcl

    DEPENDS
    MLIRHeteroCLOpsIncGen
    MLIRHeteroCLTypesIncGen
    MLIRHeteroCLPassesIncGen
    MLIRHeteroCLConversionPassesIncGen

    LINK_LIBS PUBLIC
    ${conversion_libs}
    MLIRIR
    MLIRPass
    MLIRMathTransforms
    MLIRHeteroCL
    MLIRHCLSupport
)