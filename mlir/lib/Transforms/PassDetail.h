/*
 * Copyright HeteroCL authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HCL_MLIR_PASSDETAIL_H
#define HCL_MLIR_PASSDETAIL_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace hcl {

#define GEN_PASS_CLASSES
#include "hcl/Transforms/Passes.h.inc"

} // namespace hcl
} // end namespace mlir

#endif // HCL_MLIR_PASSDETAIL_H
