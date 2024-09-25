/*
 * Copyright HeteroCL authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------------------------------------------------===//
// LowerBitOps Pass
// This file defines the lowering of bit operations.
// - GetBit
// - SetBit
// - GetSlice
// - SetSlice
// - BitReverse
//===----------------------------------------------------------------------===//

#include "hcl/Conversion/Passes.h"
#include "hcl/Dialect/HeteroCLDialect.h"
#include "hcl/Dialect/HeteroCLOps.h"
#include "hcl/Dialect/HeteroCLTypes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace mlir;
using namespace hcl;

namespace mlir {
namespace hcl {

void lowerBitReverseOps(func::FuncOp &func) {
  SmallVector<Operation *, 8> bitReverseOps;
  func.walk([&](Operation *op) {
    if (auto bitReverseOp = dyn_cast<BitReverseOp>(op)) {
      bitReverseOps.push_back(bitReverseOp);
    }
  });

  for (auto op : bitReverseOps) {
    auto bitReverseOp = dyn_cast<BitReverseOp>(op);
    Value input = bitReverseOp.getOperand();
    Location loc = bitReverseOp.getLoc();
    unsigned iwidth = input.getType().getIntOrFloatBitWidth();
    OpBuilder rewriter(bitReverseOp);
    // Create two constants: number of bits, and zero
    Value const_width_i32 = rewriter.create<mlir::arith::ConstantIntOp>(
        loc, iwidth - 1, rewriter.getI32Type());
    Value const_width = rewriter.create<mlir::arith::IndexCastOp>(
        loc, rewriter.getIndexType(), const_width_i32);
    SmallVector<Value> const_0_indices;
    const_0_indices.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));

    // Create a single-element memref to store the result
    MemRefType memRefType = MemRefType::get({1}, input.getType());
    Value resultMemRef =
        rewriter.create<mlir::memref::AllocOp>(loc, memRefType);
    // Create a loop to iterate over the bits
    SmallVector<int64_t, 1> steps(1, 1);
    SmallVector<int64_t, 1> lbs(1, 0);
    SmallVector<int64_t, 1> ubs(1, iwidth);
    affine::buildAffineLoopNest(
        rewriter, loc, lbs, ubs, steps,
        [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
          Value res = nestedBuilder.create<affine::AffineLoadOp>(
              loc, resultMemRef, const_0_indices);
          // Get the bit at the width - current position
          Value reverse_idx = nestedBuilder.create<mlir::arith::SubIOp>(
              loc, const_width, ivs[0]);
          Type one_bit_type = nestedBuilder.getIntegerType(1);
          Value bit = nestedBuilder.create<mlir::hcl::GetIntBitOp>(
              loc, one_bit_type, input, reverse_idx);
          // Set the bit at the current position
          Value new_val = nestedBuilder.create<mlir::hcl::SetIntBitOp>(
              loc, res.getType(), res, ivs[0], bit);
          nestedBuilder.create<affine::AffineStoreOp>(
              loc, new_val, resultMemRef, const_0_indices);
        });
    // Load the result from resultMemRef
    Value res = rewriter.create<affine::AffineLoadOp>(loc, resultMemRef,
                                                      const_0_indices);
    op->getResult(0).replaceAllUsesWith(res);
  }

  // remove the bitReverseOps
  std::reverse(bitReverseOps.begin(), bitReverseOps.end());
  for (auto op : bitReverseOps) {
    auto v = op->getResult(0);
    if (v.use_empty()) {
      op->erase();
    }
  }
}

void lowerSetSliceOps(func::FuncOp &func) {
  SmallVector<Operation *, 8> setSliceOps;
  func.walk([&](Operation *op) {
    if (auto setSliceOp = dyn_cast<SetIntSliceOp>(op)) {
      setSliceOps.push_back(setSliceOp);
    }
  });

  for (auto op : setSliceOps) {
    auto setSliceOp = dyn_cast<SetIntSliceOp>(op);
    ValueRange operands = setSliceOp->getOperands();
    Value input = operands[0];
    Value hi = operands[1];
    Value lo = operands[2];
    Value val = operands[3];
    Location loc = op->getLoc();
    OpBuilder rewriter(op);
    // Add 1 to hi to make it inclusive
    Type i32 = rewriter.getIntegerType(32);
    Value one_i32 = rewriter.create<mlir::arith::ConstantIntOp>(loc, 1, i32);
    Value one_idx =
        rewriter.create<mlir::arith::IndexCastOp>(loc, hi.getType(), one_i32);
    Value ub = rewriter.create<mlir::arith::AddIOp>(loc, hi, one_idx);

    // Extend val to the same width as input
    if (val.getType().getIntOrFloatBitWidth() <
        input.getType().getIntOrFloatBitWidth()) {
      val = rewriter.create<mlir::arith::ExtUIOp>(loc, input.getType(), val);
    }

    // Create a step of 1, index type
    Value step_i32 = rewriter.create<mlir::arith::ConstantIntOp>(loc, 1, i32);
    Value step =
        rewriter.create<mlir::arith::IndexCastOp>(loc, hi.getType(), step_i32);

    // build an SCF for loop to iterate over the bits
    scf::ForOp loop = rewriter.create<scf::ForOp>(
        loc, lo, ub, step, ValueRange({input}),
        [&](OpBuilder &builder, Location loc, Value iv, ValueRange ivs) {
          // get the (iv - lo)-th bit of val
          Value iv_sub_lo = builder.create<mlir::arith::SubIOp>(loc, iv, lo);
          Value idx_casted = builder.create<mlir::arith::IndexCastOp>(
              loc, val.getType(), iv_sub_lo);
          Value val_shifted =
              builder.create<mlir::arith::ShRUIOp>(loc, val, idx_casted);
          Value bit;
          if (val_shifted.getType().getIntOrFloatBitWidth() > 1) {
            Type one_bit_type = builder.getIntegerType(1);
            bit = builder.create<mlir::arith::TruncIOp>(loc, one_bit_type,
                                                        val_shifted);
          } else {
            bit = val_shifted;
          }
          Value res = builder.create<mlir::hcl::SetIntBitOp>(
              loc, ivs[0].getType(), ivs[0], iv, bit);
          builder.create<scf::YieldOp>(loc, res);
        });

    // replace the result of the setSliceOp with the result of the loop
    setSliceOp->getResult(0).replaceAllUsesWith(loop.getResult(0));
  }

  // remove the setSliceOps
  std::reverse(setSliceOps.begin(), setSliceOps.end());
  for (auto op : setSliceOps) {
    auto v = op->getResult(0);
    if (v.use_empty()) {
      op->erase();
    }
  }
}

/// Pass entry point
bool applyLowerBitOps(ModuleOp &mod) {
  for (func::FuncOp func : mod.getOps<func::FuncOp>()) {
    lowerBitReverseOps(func);
    lowerSetSliceOps(func);
  }
  return true;
}
} // namespace hcl
} // namespace mlir

namespace {
struct HCLLowerBitOpsTransformation
    : public LowerBitOpsBase<HCLLowerBitOpsTransformation> {
  void runOnOperation() override {
    auto mod = getOperation();
    if (!applyLowerBitOps(mod)) {
      return signalPassFailure();
    }
  }
};
} // namespace

namespace mlir {
namespace hcl {

std::unique_ptr<OperationPass<ModuleOp>> createLowerBitOpsPass() {
  return std::make_unique<HCLLowerBitOpsTransformation>();
}
} // namespace hcl
} // namespace mlir