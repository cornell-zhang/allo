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
    SmallVector<ViewWithLayoutOp, 8> setViewWithLayoutOps;
    func.walk([&](Operation *op) {
      if (auto viewWithLayoutOp = dyn_cast<ViewWithLayoutOp>(op)) {
        setViewWithLayoutOps.push_back(viewWithLayoutOp);
      }
    });
    for (auto op : setViewWithLayoutOps) {
      Value input = op->getOperands()[0];
      Value output = op->getResults()[0];
      MemRefType memRefType = output.getType().dyn_cast<MemRefType>();
      auto result_shape = memRefType.getShape();
      auto offsets = op.getOffsets();
      auto sizes = op.getSizes();
      auto strides = op.getStrides();
      if (offsets.size() != sizes.size() || sizes.size() != strides.size()) {
        return false;
      }
      int64_t expected_size = 1, viewed_size = 1;
      for (int64_t val : result_shape) {
        expected_size *= val;
      }
      for (int64_t val : sizes) {
        viewed_size *= val;
      }
      if (expected_size != viewed_size) {
        return false;
      }
      // lower to load-store
      Location loc = op->getLoc();
      OpBuilder rewriter(op);
      Type eltTy = memRefType.getElementType();
      auto flatType = MemRefType::get({viewed_size}, eltTy);
      Value flatAlloc = rewriter.create<memref::AllocOp>(loc, flatType);
      SmallVector<OpFoldResult> dimAttr, strideAttr;
      dimAttr.push_back(rewriter.getIndexAttr(viewed_size));
      strideAttr.push_back(rewriter.getIndexAttr(1));
      Value flatInput = rewriter.create<memref::ReinterpretCastOp>(
          loc, flatType, input, rewriter.getIndexAttr(0), dimAttr, strideAttr);
      // memory access with viewed layout
      SmallVector<int64_t> lbs(sizes.size(), 0),
          ubs(sizes.begin(), sizes.end()), steps(sizes.size(), 1);
      SmallVector<int64_t> dst_strides;
      int64_t stride = expected_size;
      for (size_t i = 0; i < sizes.size(); ++i) {
        stride /= sizes[i];
        dst_strides.push_back(stride);
      }
      affine::buildAffineLoopNest(
          rewriter, loc, lbs, sizes, steps,
          [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange ivs) {
            Value srcIdx =
                nestedBuilder.create<arith::ConstantIndexOp>(nestedLoc, 0);
            Value dstIdx =
                nestedBuilder.create<arith::ConstantIndexOp>(nestedLoc, 0);
            for (unsigned d = 0; d < sizes.size(); ++d) {
              Value add = nestedBuilder.create<arith::AddIOp>(
                  nestedLoc, ivs[d],
                  nestedBuilder.create<arith::ConstantIndexOp>(nestedLoc,
                                                               offsets[d]));
              Value mul = nestedBuilder.create<arith::MulIOp>(
                  nestedLoc, add,
                  nestedBuilder.create<arith::ConstantIndexOp>(nestedLoc,
                                                               strides[d]));
              srcIdx =
                  nestedBuilder.create<arith::AddIOp>(nestedLoc, srcIdx, mul);
              mul = nestedBuilder.create<arith::MulIOp>(
                  nestedLoc, ivs[d],
                  nestedBuilder.create<arith::ConstantIndexOp>(nestedLoc,
                                                               dst_strides[d]));
              dstIdx =
                  nestedBuilder.create<arith::AddIOp>(nestedLoc, dstIdx, mul);
            }
            Value val = nestedBuilder.create<memref::LoadOp>(
                nestedLoc, flatInput, ValueRange{srcIdx});
            nestedBuilder.create<memref::StoreOp>(nestedLoc, val, flatAlloc,
                                                  ValueRange{dstIdx});
          });
      SmallVector<OpFoldResult> outputDimAttr, outputStrideAttr;
      stride = expected_size;
      for (size_t i = 0; i < result_shape.size(); ++i) {
        outputDimAttr.push_back(rewriter.getIndexAttr(result_shape[i]));
        stride /= result_shape[i];
        outputStrideAttr.push_back(rewriter.getIndexAttr(stride));
      }
      Value reshapedOutput = rewriter.create<memref::ReinterpretCastOp>(
          loc, memRefType, flatAlloc, rewriter.getIndexAttr(0), outputDimAttr,
          outputStrideAttr);
      output.replaceAllUsesWith(reshapedOutput);
      op->erase();
    }
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