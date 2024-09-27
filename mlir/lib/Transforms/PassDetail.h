/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef Allo_MLIR_PASSDETAIL_H
#define Allo_MLIR_PASSDETAIL_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace allo {

#define GEN_PASS_CLASSES
#include "allo/Transforms/Passes.h.inc"

} // namespace allo
} // end namespace mlir

#endif // Allo_MLIR_PASSDETAIL_H
