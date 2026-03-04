/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/InitAllDialects.h"
#include "allo/InitAllExtensions.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

using namespace mlir;

int main(int argc, char **argv) {
  DialectRegistry registry;
  allo::registerAllDialects(registry);
  allo::registerAllExtensions(registry);
  return asMainReturnCode(
      MlirOptMain(argc, argv, "Allo optimization driver\n", registry));
}