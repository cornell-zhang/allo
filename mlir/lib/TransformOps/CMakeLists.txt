# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

add_mlir_library(MLIRAlloTransformOps
        AlloTransformOps.cpp

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