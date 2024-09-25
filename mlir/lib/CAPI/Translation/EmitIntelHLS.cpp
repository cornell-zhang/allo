/*
 * Copyright HeteroCL authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hcl/Translation/EmitIntelHLS.h"
#include "hcl-c/Translation/EmitIntelHLS.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/CAPI/Utils.h"

using namespace mlir;
using namespace hcl;

MlirLogicalResult mlirEmitIntelHls(MlirModule module,
                                   MlirStringCallback callback,
                                   void *userData) {
  mlir::detail::CallbackOstream stream(callback, userData);
  return wrap(emitIntelHLS(unwrap(module), stream));
}