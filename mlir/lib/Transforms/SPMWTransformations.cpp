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
 * Compile-time loop unrolling for meta loops.
 *
 * This pass scans the IR and fully unrolls loops that are annotated with the
 * attribute `loop_type = "unroll"`. Such loops are intended to be compile-time
 * constructs (e.g., meta_for), and therefore should be completely expanded in
 * the IR.
 */
bool applyUnrollMetaFor(Operation *func) {
  func->walk<WalkOrder::PostOrder>([&](AffineForOp forOp) {
    auto attr = forOp->getAttrOfType<StringAttr>("loop_type");
    if (attr && attr.getValue() == "unroll") {
      if (failed(loopUnrollFull(forOp))) {
        forOp.emitError("failed to fully unroll the loop");
      }
    }
  });
  return true;
}

} // namespace allo
} // namespace mlir
