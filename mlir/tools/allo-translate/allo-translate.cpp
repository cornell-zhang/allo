/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------------------------------------------------===//
//
// This is a command line utility that translates a file from/to MLIR using one
// of the registered translations.
//
//===----------------------------------------------------------------------===//

#include "allo/Translation/EmitIntelHLS.h"
#include "allo/Translation/EmitVivadoHLS.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#ifdef OPENSCOP
#include "allo/Target/OpenSCoP/ExtractScopStmt.h"
#endif

#include "allo/Dialect/AlloDialect.h"

int main(int argc, char **argv) {
  mlir::registerAllTranslations();
  mlir::allo::registerEmitVivadoHLSTranslation();
  mlir::allo::registerEmitIntelHLSTranslation();
#ifdef OPENSCOP
  mlir::allo::registerToOpenScopExtractTranslation();
#endif

  return failed(mlir::mlirTranslateMain(
      argc, argv, "Allo MLIR Dialect Translation Tool"));
}
