/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Translation/EmitTapaHLS.h"
#include "allo-c/Translation/EmitTapaHLS.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/CAPI/Utils.h"

using namespace mlir;
using namespace allo;

MlirLogicalResult mlirEmitTapaHls(MlirModule module,
                                  MlirStringCallback callback, void *userData) {
  mlir::detail::CallbackOstream stream(callback, userData);
  return wrap(emitTapaHLS(unwrap(module), stream));
}