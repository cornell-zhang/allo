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
  for (func::FuncOp func : mod.getOps<func::FuncOp>()) {
    SmallVector<StoreSliceOp, 8> setStoreSliceOps;
    func.walk([&](Operation *op) {
      if (auto storeSliceOp = dyn_cast<StoreSliceOp>(op)) {
        setStoreSliceOps.push_back(storeSliceOp);
      }
    });
    for (auto op : setStoreSliceOps) {
      Value dst = op->getOperands()[1];
      Value tile = op->getOperands()[0];
      MemRefType dstType = dst.getType().dyn_cast<MemRefType>();
      MemRefType tileType = tile.getType().dyn_cast<MemRefType>();
      auto tile_shape = tileType.getShape();
      auto offsets = op.getOffsets();
      auto sizes = op.getSizes();
      auto strides = op.getStrides();
      if (offsets.size() != sizes.size() || sizes.size() != strides.size()) {
        return false;
      }
      int64_t tile_size = 1, slice_size = 1;
      for (int64_t val : tile_shape) {
        tile_size *= val;
      }
      for (int64_t val : sizes) {
        slice_size *= val;
      }
      if (tile_size != slice_size) {
        return false;
      }
      // lower to load-store
      Location loc = op->getLoc();
      OpBuilder rewriter(op);
      // flatten the dst memref
      int64_t flattened_size = 1;
      for (int64_t val : dstType.getShape()) {
        flattened_size *= val;
      }
      auto tileFlatType =
          MemRefType::get({tile_size}, dstType.getElementType());
      SmallVector<OpFoldResult> tileDimAttr, tileStrideAttr;
      tileDimAttr.push_back(rewriter.getIndexAttr(tile_size));
      tileStrideAttr.push_back(rewriter.getIndexAttr(1));
      Value flatTile = rewriter.create<memref::ReinterpretCastOp>(
          loc, tileFlatType, tile, rewriter.getIndexAttr(0), tileDimAttr,
          tileStrideAttr);
      auto flatType =
          MemRefType::get({flattened_size}, dstType.getElementType());
      SmallVector<OpFoldResult> dimAttr, strideAttr;
      dimAttr.push_back(rewriter.getIndexAttr(flattened_size));
      strideAttr.push_back(rewriter.getIndexAttr(1));
      Value flatDst = rewriter.create<memref::ReinterpretCastOp>(
          loc, flatType, dst, rewriter.getIndexAttr(0), dimAttr, strideAttr);
      // element-wise store
      SmallVector<int64_t> lbs(sizes.size(), 0), steps(sizes.size(), 1);
      SmallVector<int64_t> src_strides;
      int64_t stride = tile_size;
      for (size_t i = 0; i < sizes.size(); ++i) {
        stride /= sizes[i];
        src_strides.push_back(stride);
      }
      affine::buildAffineLoopNest(
          rewriter, loc, lbs, sizes, steps,
          [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange ivs) {
            Value srcIdx =
                nestedBuilder.create<arith::ConstantIndexOp>(nestedLoc, 0);
            Value dstIdx =
                nestedBuilder.create<arith::ConstantIndexOp>(nestedLoc, 0);
            for (unsigned d = 0; d < sizes.size(); ++d) {
              Value add = nestedBuilder.create<arith::AddIOp>(nestedLoc, ivs[d],
                                                              offsets[d]);
              Value mul = nestedBuilder.create<arith::MulIOp>(
                  nestedLoc, add,
                  nestedBuilder.create<arith::ConstantIndexOp>(nestedLoc,
                                                               strides[d]));
              dstIdx =
                  nestedBuilder.create<arith::AddIOp>(nestedLoc, dstIdx, mul);
              mul = nestedBuilder.create<arith::MulIOp>(
                  nestedLoc, ivs[d],
                  nestedBuilder.create<arith::ConstantIndexOp>(nestedLoc,
                                                               src_strides[d]));
              srcIdx =
                  nestedBuilder.create<arith::AddIOp>(nestedLoc, srcIdx, mul);
            }
            // load from tile
            Value elem = nestedBuilder.create<memref::LoadOp>(
                nestedLoc, flatTile, ValueRange{srcIdx});
            // store to the flattened memref
            nestedBuilder.create<memref::StoreOp>(nestedLoc, elem, flatDst,
                                                  ValueRange{dstIdx});
          });
      op->erase();
    }
  }
  return true;
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
  for (func::FuncOp func : mod.getOps<func::FuncOp>()) {
    SmallVector<LoadSliceOp, 8> setLoadSliceOps;
    func.walk([&](Operation *op) {
      if (auto loadSliceOp = dyn_cast<LoadSliceOp>(op)) {
        setLoadSliceOps.push_back(loadSliceOp);
      }
    });
    for (auto op : setLoadSliceOps) {
      // TODO
      return false;
    }
  }
  return true;
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