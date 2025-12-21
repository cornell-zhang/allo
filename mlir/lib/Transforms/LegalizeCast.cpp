/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PassDetail.h"

#include "allo/Dialect/AlloDialect.h"
#include "allo/Dialect/AlloOps.h"
#include "allo/Dialect/AlloTypes.h"
#include "allo/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace mlir;
using namespace allo;

namespace mlir {
namespace allo {

void legalizeCast(func::FuncOp &func) {
  SmallVector<Operation *, 8> IntToFPOps;
  func.walk([&](Operation *op) {
    if (auto intToFloatCastOp = dyn_cast<arith::SIToFPOp>(op)) {
      IntToFPOps.push_back(intToFloatCastOp);
    } else if (auto uintToFloatCastOp = dyn_cast<arith::UIToFPOp>(op)) {
      IntToFPOps.push_back(uintToFloatCastOp);
    }
  });

  for (auto op : IntToFPOps) {
    auto input = op->getOperand(0);
    auto res = op->getResult(0);
    Location loc = op->getLoc();
    OpBuilder rewriter(op);
    size_t twidth = res.getType().getIntOrFloatBitWidth();   // target width
    size_t iwidth = input.getType().getIntOrFloatBitWidth(); // input width
    bool isSigned;
    if (auto intToFloatCastOp = dyn_cast<arith::SIToFPOp>(op)) {
      isSigned = true;
    } else if (auto uintToFloatCastOp = dyn_cast<arith::UIToFPOp>(op)) {
      isSigned = false;
    } else {
      llvm_unreachable("unexpected op");
    }

    Type targetIntType = IntegerType::get(op->getContext(), twidth);
    if (iwidth > twidth) {
      // If the input is wider than the target, we need to truncate it.
      Value truncated =
          rewriter.create<arith::TruncIOp>(loc, targetIntType, input);
      op->setOperand(0, truncated);
    } else if (iwidth < twidth) {
      // If the input is narrower than the target, we need to extend it.
      if (isSigned) {
        Value extended =
            rewriter.create<arith::ExtSIOp>(loc, targetIntType, input);
        op->setOperand(0, extended);
      } else {
        Value extended =
            rewriter.create<arith::ExtUIOp>(loc, targetIntType, input);
        op->setOperand(0, extended);
      }
    } else {
      continue; // No legalization needed
    }
  }
}

/// Pass entry point
bool applyLegalizeCast(ModuleOp &module) {
  for (func::FuncOp func : module.getOps<func::FuncOp>()) {
    legalizeCast(func);
  }
  return true;
}

} // namespace allo
} // namespace mlir

namespace {
struct AlloLegalizeCastTransformation
    : public mlir::allo::impl::LegalizeCastBase<
          AlloLegalizeCastTransformation> {
  void runOnOperation() override {
    auto mod = getOperation();
    if (!applyLegalizeCast(mod)) {
      signalPassFailure();
    }
  }
};
} // namespace

namespace mlir {
namespace allo {
std::unique_ptr<OperationPass<ModuleOp>> createLegalizeCastPass() {
  return std::make_unique<AlloLegalizeCastTransformation>();
}
} // namespace allo
} // namespace mlir