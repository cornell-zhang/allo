/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Translation/EmitCatapultHLS.h"
#include "allo-c/Translation/EmitCatapultHLS.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/CAPI/Utils.h"

using namespace mlir;
using namespace allo;

MlirLogicalResult mlirEmitCatapultHls(MlirModule module,
                                      MlirStringCallback callback,
                                      void *userData) {
  mlir::detail::CallbackOstream stream(callback, userData);
  return wrap(emitCatapultHLS(unwrap(module), stream));
}