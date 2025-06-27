/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------------------------------------------------===//
// Lower ViewWithLayout Ops
//===----------------------------------------------------------------------===//
#include "allo/Conversion/Passes.h"
#include "allo/Dialect/AlloDialect.h"
#include "allo/Dialect/AlloOps.h"
#include "allo/Dialect/AlloTypes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace mlir;
using namespace allo;

namespace mlir {
namespace allo {

/// Pass entry point
bool applyLowerViewWithLayoutOps(ModuleOp &mod) {
  for (func::FuncOp func : mod.getOps<func::FuncOp>()) {
    // TODO
  }
  return true;
}
} // namespace allo
} // namespace mlir

namespace {
struct AlloLowerViewWithLayoutOpsTransformation
    : public LowerViewWithLayoutOpsBase<
          AlloLowerViewWithLayoutOpsTransformation> {
  void runOnOperation() override {
    auto mod = getOperation();
    if (!applyLowerViewWithLayoutOps(mod)) {
      return signalPassFailure();
    }
  }
};
} // namespace

namespace mlir {
namespace allo {

std::unique_ptr<OperationPass<ModuleOp>> createLowerViewWithLayoutOpsPass() {
  return std::make_unique<AlloLowerViewWithLayoutOpsTransformation>();
}
} // namespace allo
} // namespace mlir