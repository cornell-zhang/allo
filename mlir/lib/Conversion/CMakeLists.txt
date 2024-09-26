# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

file(GLOB globbed *.cpp)

get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_mlir_dialect_library(MLIRAlloConversion
    ${globbed}

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/allo

    DEPENDS
    MLIRAlloOpsIncGen
    MLIRAlloTypesIncGen
    MLIRAlloPassesIncGen
    MLIRAlloConversionPassesIncGen

    LINK_LIBS PUBLIC
    ${conversion_libs}
    MLIRIR
    MLIRPass
    MLIRMathTransforms
    MLIRAllo
    MLIRAlloSupport
)