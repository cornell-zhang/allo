/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_ANALYSIS_LIVENESS_H
#define ALLO_ANALYSIS_LIVENESS_H
#include "allo/Dialect/AlloDialect.h"
#include "allo/Dialect/AlloOps.h"
#include "allo/Dialect/AlloTypes.h"
#include "mlir/CAPI/IR.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace mlir;
using namespace allo;
namespace mlir {
namespace allo {
Operation *getFirstUse(Value value, Operation &func);
Operation *getLastUse(Value value, Operation &func);
Operation *getNextUse(Value value, Operation *curUse, Operation &func);
} // namespace allo
} // namespace mlir

#endif // ALLO_ANALYSIS_LIVENESS_H
