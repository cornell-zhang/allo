# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

add_mlir_dialect_library(MLIRAllo
        AlloDialect.cpp
        AlloOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/allo

        DEPENDS
        MLIRAlloOpsIncGen
        MLIRAlloTypesIncGen
        MLIRAlloAttrsIncGen
        MLIRAlloEnumsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
	)
