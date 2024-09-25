# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

add_mlir_public_c_api_library(MLIRHCLCAPIEmitHLSCpp
  EmitVivadoHLS.cpp
  EmitIntelHLS.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir-c

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRHCLEmitHLSCpp
  )
