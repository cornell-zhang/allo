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
      auto tileFlatType =
          MemRefType::get({tile_size}, dstType.getElementType());
      SmallVector<OpFoldResult> tileDimAttr, tileStrideAttr;
      tileDimAttr.push_back(rewriter.getIndexAttr(tile_size));
      tileStrideAttr.push_back(rewriter.getIndexAttr(1));
      Value flatTile = rewriter.create<memref::ReinterpretCastOp>(
          loc, tileFlatType, tile, rewriter.getIndexAttr(0), tileDimAttr,
          tileStrideAttr);
      int64_t flattened_size = 1;
      for (int64_t val : dstType.getShape()) {
        flattened_size *= val;
      }
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
      Value src = op->getOperands()[0];
      Value tile = op->getResults()[0];
      MemRefType srcType = src.getType().dyn_cast<MemRefType>();
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
      // flatten the memref
      auto tileFlatType =
          MemRefType::get({tile_size}, tileType.getElementType());
      Value flatTile = rewriter.create<memref::AllocOp>(loc, tileFlatType);
      int64_t flattened_size = 1;
      for (int64_t val : srcType.getShape()) {
        flattened_size *= val;
      }
      auto flatType =
          MemRefType::get({flattened_size}, tileType.getElementType());
      SmallVector<OpFoldResult> dimAttr, strideAttr;
      dimAttr.push_back(rewriter.getIndexAttr(flattened_size));
      strideAttr.push_back(rewriter.getIndexAttr(1));
      Value flatSrc = rewriter.create<memref::ReinterpretCastOp>(
          loc, flatType, src, rewriter.getIndexAttr(0), dimAttr, strideAttr);
      // element-wise load
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
              srcIdx =
                  nestedBuilder.create<arith::AddIOp>(nestedLoc, srcIdx, mul);
              mul = nestedBuilder.create<arith::MulIOp>(
                  nestedLoc, ivs[d],
                  nestedBuilder.create<arith::ConstantIndexOp>(nestedLoc,
                                                               src_strides[d]));
              dstIdx =
                  nestedBuilder.create<arith::AddIOp>(nestedLoc, dstIdx, mul);
            }
            // load from tile
            Value elem = nestedBuilder.create<memref::LoadOp>(
                nestedLoc, flatSrc, ValueRange{srcIdx});
            // store to the flattened memref
            nestedBuilder.create<memref::StoreOp>(nestedLoc, elem, flatTile,
                                                  ValueRange{dstIdx});
          });
      SmallVector<OpFoldResult> tileDimAttr, tileStrideAttr;
      stride = tile_size;
      for (size_t i = 0; i < tile_shape.size(); ++i) {
        tileDimAttr.push_back(rewriter.getIndexAttr(tile_shape[i]));
        stride /= tile_shape[i];
        tileStrideAttr.push_back(rewriter.getIndexAttr(stride));
      }
      Value reshapedTile = rewriter.create<memref::ReinterpretCastOp>(
          loc, tileType, flatTile, rewriter.getIndexAttr(0), tileDimAttr,
          tileStrideAttr);
      tile.replaceAllUsesWith(reshapedTile);
      op->erase();
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

//===----------------------------------------------------------------------===//
// Lower TransformLayout Ops
//===----------------------------------------------------------------------===//

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
      MemRefType memRefType = output.getType().dyn_cast<MemRefType>();
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
      SmallVector<int64_t> lbs(sizes.size(), 0), steps(sizes.size(), 1);
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
    : public LowerTransformLayoutOpsBase<
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

std::unique_ptr<OperationPass<ModuleOp>> createLowerStoreSliceOpsPass() {
  return std::make_unique<AlloLowerStoreSliceOpsTransformation>();
}

std::unique_ptr<OperationPass<ModuleOp>> createLowerLoadSliceOpsPass() {
  return std::make_unique<AlloLowerLoadSliceOpsTransformation>();
}

std::unique_ptr<OperationPass<ModuleOp>> createLowerTransformLayoutOpsPass() {
  return std::make_unique<AlloLowerTransformLayoutOpsTransformation>();
}
} // namespace allo
} // namespace mlir