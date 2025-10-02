/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

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

//===----------------------------------------------------------------------===//
// Lower Store Memref Slice Ops
//===----------------------------------------------------------------------===//
namespace mlir {
namespace allo {
bool applyLowerStoreSliceOps(ModuleOp &mod) {
  // TODO
  return false;
}
} // namespace allo
} // namespace mlir

namespace {
struct AlloLowerStoreSliceOpsTransformation
    : public LowerStoreSliceOpsBase<AlloLowerStoreSliceOpsTransformation> {
  void runOnOperation() override {
    auto mod = getOperation();
    if (!applyLowerStoreSliceOps(mod)) {
      return signalPassFailure();
    }
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Lower Load Memref Slice Ops
//===----------------------------------------------------------------------===//
namespace mlir {
namespace allo {
bool applyLowerLoadSliceOps(ModuleOp &mod) {
  // TODO
  return false;
}
} // namespace allo
} // namespace mlir

namespace {
struct AlloLowerLoadSliceOpsTransformation
    : public LowerLoadSliceOpsBase<AlloLowerLoadSliceOpsTransformation> {
  void runOnOperation() override {
    auto mod = getOperation();
    if (!applyLowerLoadSliceOps(mod)) {
      return signalPassFailure();
    }
  }
};
} // namespace

namespace mlir {
namespace allo {

std::unique_ptr<OperationPass<ModuleOp>> createLowerStoreSliceOpsPass() {
  return std::make_unique<AlloLowerStoreSliceOpsTransformation>();
}

std::unique_ptr<OperationPass<ModuleOp>> createLowerLoadSliceOpsPass() {
  return std::make_unique<AlloLowerLoadSliceOpsTransformation>();
}
} // namespace allo
} // namespace mlir