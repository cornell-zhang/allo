/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Translation/EmitXlsHLS.h"
#include "allo-c/Translation/EmitXlsHLS.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/CAPI/Utils.h"

using namespace mlir;
using namespace allo;

MlirLogicalResult mlirEmitXlsHls(MlirModule module, MlirStringCallback callback,
                                 void *userData, bool useMemory) {
  mlir::detail::CallbackOstream stream(callback, userData);
  return wrap(emitXlsHLS(unwrap(module), stream, useMemory));
}
