/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------------------------------------------------===//
// Lower MemCopy Ops
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
#define GEN_PASS_DEF_LOWERMEMCOPYOPS
#include "allo/Conversion/Passes.h.inc"
} // namespace allo
} // namespace mlir

namespace mlir {
namespace allo {

/// Pass entry point
bool applyLowerMemCopyOps(ModuleOp &mod) {
  for (func::FuncOp func : mod.getOps<func::FuncOp>()) {
    SmallVector<memref::CopyOp, 8> copyOps;
    func.walk([&](memref::CopyOp op) { copyOps.push_back(op); });

    for (auto op : copyOps) {
      Value source = op.getSource();
      Value target = op.getTarget();
      MemRefType srcType = llvm::dyn_cast<MemRefType>(source.getType());
      MemRefType dstType = llvm::dyn_cast<MemRefType>(target.getType());

      if (!srcType || !dstType) {return false;}

      auto srcShape = srcType.getShape();
      auto dstShape = dstType.getShape();

      // Ensure element count matches
      int64_t srcSize = 1;
      for (int64_t val : srcShape)
        srcSize *= val;

      int64_t dstSize = 1;
      for (int64_t val : dstShape)
        dstSize *= val;

      if (srcSize != dstSize) {return false;}

      OpBuilder rewriter(op);
      Location loc = op.getLoc();

      if (srcShape == dstShape) {
        // Optimization for matching shapes
        SmallVector<int64_t> lbs(dstShape.size(), 0);
        SmallVector<int64_t> ubs(dstShape.begin(), dstShape.end());
        SmallVector<int64_t> steps(dstShape.size(), 1);

        affine::buildAffineLoopNest(
            rewriter, loc, lbs, ubs, steps,
            [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange ivs) {
              Value val = nestedBuilder.create<memref::LoadOp>(nestedLoc, source, ivs);
              nestedBuilder.create<memref::StoreOp>(nestedLoc, val, target, ivs);
            });
      } else {
        // General case: flat loops with linear indices mapped from target shape
        Type eltTy = srcType.getElementType();
        auto flatType = MemRefType::get({dstSize}, eltTy);

        SmallVector<OpFoldResult> dimAttr, strideAttr;
        dimAttr.push_back(rewriter.getIndexAttr(dstSize));
        strideAttr.push_back(rewriter.getIndexAttr(1));

        Value flatSrc = rewriter.create<memref::ReinterpretCastOp>(
            loc, flatType, source, rewriter.getIndexAttr(0), dimAttr,
            strideAttr);
        Value flatDst = rewriter.create<memref::ReinterpretCastOp>(
            loc, flatType, target, rewriter.getIndexAttr(0), dimAttr,
            strideAttr);

        SmallVector<int64_t> lbs(dstShape.size(), 0);
        SmallVector<int64_t> ubs(dstShape.begin(), dstShape.end());
        SmallVector<int64_t> steps(dstShape.size(), 1);

        SmallVector<int64_t> dst_strides;
        int64_t stride = dstSize;
        for (size_t i = 0; i < dstShape.size(); ++i) {
          stride /= dstShape[i];
          dst_strides.push_back(stride);
        }

        affine::buildAffineLoopNest(
            rewriter, loc, lbs, ubs, steps,
            [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange ivs) {
              Value dstIdx =
                  nestedBuilder.create<arith::ConstantIndexOp>(nestedLoc, 0);

              for (unsigned d = 0; d < dstShape.size(); ++d) {
                Value mul = nestedBuilder.create<arith::MulIOp>(
                    nestedLoc, ivs[d],
                    nestedBuilder.create<arith::ConstantIndexOp>(
                        nestedLoc, dst_strides[d]));
                dstIdx =
                    nestedBuilder.create<arith::AddIOp>(nestedLoc, dstIdx, mul);
              }

              // Load using flat linear index from flat source
              Value val = nestedBuilder.create<memref::LoadOp>(nestedLoc, flatSrc, ValueRange{dstIdx});
              // Store using flat linear index into flat target
              nestedBuilder.create<memref::StoreOp>(nestedLoc, val, flatDst, ValueRange{dstIdx});
            });
      }
      
      op->erase();
    }
  }
  return true;
}

} // namespace allo
} // namespace mlir

namespace {
struct AlloLowerMemCopyOpsTransformation
    : public mlir::allo::impl::LowerMemCopyOpsBase<
          AlloLowerMemCopyOpsTransformation> {
  void runOnOperation() override {
    auto mod = getOperation();
    if (!applyLowerMemCopyOps(mod)) {
      return signalPassFailure();
    }
  }
};
} // namespace

namespace mlir {
namespace allo {

std::unique_ptr<OperationPass<ModuleOp>> createLowerMemCopyOpsPass() {
  return std::make_unique<AlloLowerMemCopyOpsTransformation>();
}

} // namespace allo
} // namespace mlir
