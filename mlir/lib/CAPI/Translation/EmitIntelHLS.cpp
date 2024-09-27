/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Translation/EmitIntelHLS.h"
#include "allo-c/Translation/EmitIntelHLS.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/CAPI/Utils.h"

using namespace mlir;
using namespace allo;

MlirLogicalResult mlirEmitIntelHls(MlirModule module,
                                   MlirStringCallback callback,
                                   void *userData) {
  mlir::detail::CallbackOstream stream(callback, userData);
  return wrap(emitIntelHLS(unwrap(module), stream));
}