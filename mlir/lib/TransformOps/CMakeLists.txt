# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

add_mlir_library(MLIRAlloTransformOps
    DialectExtension.cpp
    RaiseToAffine.cpp
    LoopTransformations.cpp
    Utils.cpp

    DEPENDS
    MLIRAlloIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRAffineDialect
    MLIRAffineUtils
    MLIRFuncDialect
    MLIRMemRefDialect
    MLIRSCFDialect
    MLIRTransformDialect
)