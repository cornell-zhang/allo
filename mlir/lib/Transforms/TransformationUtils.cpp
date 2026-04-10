/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PassDetail.h"
#include "allo/Dialect/AlloDialect.h"
#include "allo/Dialect/AlloOps.h"
#include "allo/Support/Utils.h"
#include "allo/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"

using namespace mlir;
using namespace allo;

namespace mlir {
namespace allo {

/*
 * Unroll affine loop with given factor
 */
bool applyUnroll(Operation *op, int64_t factor) {
  auto loop = dyn_cast<AffineForOp>(op);
  if (!loop) {
    loop.emitError("Expect to unroll an affine.for operation");
    return false;
  }
  if (factor > 0) {
    return succeeded(loopUnrollUpToFactor(loop, factor));
  } else {
    return succeeded(loopUnrollFull(loop));
  }
  return true;
}

} // namespace allo
} // namespace mlir
