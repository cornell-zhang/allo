# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

add_mlir_library(MLIRAlloPasses
    LoopTransformations.cpp
    AnyWidthInteger.cpp
    MoveReturnToInput.cpp
    Passes.cpp
    LegalizeCast.cpp
    RemoveStrideMap.cpp
    MemRefDCE.cpp
    DataPlacement.cpp
    UnifyKernels.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/allo

    DEPENDS
    MLIRAlloOpsIncGen
    MLIRAlloTypesIncGen
    MLIRAlloPassesIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRAllo
    MLIRAlloSupport
)