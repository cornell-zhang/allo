/*
 * Copyright HeteroCL authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------------------------------------------------===//
//
// This is a command line utility that translates a file from/to MLIR using one
// of the registered translations.
//
//===----------------------------------------------------------------------===//

#include "hcl/Translation/EmitIntelHLS.h"
#include "hcl/Translation/EmitVivadoHLS.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#ifdef OPENSCOP
#include "hcl/Target/OpenSCoP/ExtractScopStmt.h"
#endif

#include "hcl/Dialect/HeteroCLDialect.h"

int main(int argc, char **argv) {
  mlir::registerAllTranslations();
  mlir::hcl::registerEmitVivadoHLSTranslation();
  mlir::hcl::registerEmitIntelHLSTranslation();
#ifdef OPENSCOP
  mlir::hcl::registerToOpenScopExtractTranslation();
#endif

  return failed(mlir::mlirTranslateMain(
      argc, argv, "HeteroCL MLIR Dialect Translation Tool"));
}
