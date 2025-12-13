/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------------------------------------------------===//
// Lower TransformLayout Ops
//===----------------------------------------------------------------------===//
#include "PassDetail.h"
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
#define GEN_PASS_DEF_LOWERTRANSFORMLAYOUTOPS
#include "allo/Conversion/Passes.h.inc"
} // namespace allo
} // namespace mlir

namespace mlir {
namespace allo {

/// Pass entry point
bool applyLowerTransformLayoutOps(ModuleOp &mod) {
  for (func::FuncOp func : mod.getOps<func::FuncOp>()) {
    SmallVector<TransformLayoutOp, 8> setTransformLayoutOps;
    func.walk([&](Operation *op) {
      if (auto transformLayoutOp = dyn_cast<TransformLayoutOp>(op)) {
        setTransformLayoutOps.push_back(transformLayoutOp);
      }
    });
    for (auto op : setTransformLayoutOps) {
      Value input = op->getOperands()[0];
      Value output = op->getResults()[0];
      MemRefType memRefType = llvm::dyn_cast<MemRefType>(output.getType());
      auto result_shape = memRefType.getShape();
      auto offsets = op.getOffsets();
      auto sizes = op.getSizes();
      auto strides = op.getStrides();
      if (offsets.size() != sizes.size() || sizes.size() != strides.size()) {
        return false;
      }
      int64_t expected_size = 1, transformed_size = 1;
      for (int64_t val : result_shape) {
        expected_size *= val;
      }
      for (int64_t val : sizes) {
        transformed_size *= val;
      }
      if (expected_size != transformed_size) {
        return false;
      }
      // lower to load-store
      Location loc = op->getLoc();
      OpBuilder rewriter(op);
      Type eltTy = memRefType.getElementType();
      auto flatType = MemRefType::get({transformed_size}, eltTy);
      Value flatAlloc = rewriter.create<memref::AllocOp>(loc, flatType);
      SmallVector<OpFoldResult> dimAttr, strideAttr;
      dimAttr.push_back(rewriter.getIndexAttr(transformed_size));
      strideAttr.push_back(rewriter.getIndexAttr(1));
      Value flatInput = rewriter.create<memref::ReinterpretCastOp>(
          loc, flatType, input, rewriter.getIndexAttr(0), dimAttr, strideAttr);
      // memory access with transformed layout
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
struct AlloLowerTransformLayoutOpsTransformation
    : public mlir::allo::impl::LowerTransformLayoutOpsBase<
          AlloLowerTransformLayoutOpsTransformation> {
  void runOnOperation() override {
    auto mod = getOperation();
    if (!applyLowerTransformLayoutOps(mod)) {
      return signalPassFailure();
    }
  }
};
} // namespace

namespace mlir {
namespace allo {

std::unique_ptr<OperationPass<ModuleOp>> createLowerTransformLayoutOpsPass() {
  return std::make_unique<AlloLowerTransformLayoutOpsTransformation>();
}
} // namespace allo
} // namespace mlir