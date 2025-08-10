/*
 * Copyright Allo authors. All Rights Reserved.
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

#include "allo/Conversion/Passes.h"
#include "allo/Dialect/AlloDialect.h"
#include "allo/Dialect/AlloOps.h"
#include "allo/Dialect/AlloTypes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace mlir;
using namespace allo;

namespace mlir {
namespace allo {

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
          Value bit = nestedBuilder.create<mlir::allo::GetIntBitOp>(
              loc, one_bit_type, input, reverse_idx);
          // Set the bit at the current position
          Value new_val = nestedBuilder.create<mlir::allo::SetIntBitOp>(
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
          Value res = builder.create<mlir::allo::SetIntBitOp>(
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

void lowerSetBitOps(func::FuncOp &func) {
  SmallVector<Operation *, 8> setBitOps;
  func.walk([&](Operation *op) {
    if (auto setBitOp = dyn_cast<SetIntBitOp>(op)) {
      setBitOps.push_back(setBitOp);
    }
  });

  for (auto op : setBitOps) {
    auto setBitOp = dyn_cast<SetIntBitOp>(op);
    Value input = setBitOp.getOperand(0);
    Value index = setBitOp.getOperand(1);
    Value val = setBitOp.getOperand(2);
    Location loc = op->getLoc();
    OpBuilder rewriter(op);

    // Cast val to the same width as input
    unsigned width = input.getType().getIntOrFloatBitWidth();
    Value const_1 = rewriter.create<mlir::arith::ConstantIntOp>(loc, 1, width);

    // Cast index to i32
    Type itype = rewriter.getIntegerType(width);
    Value idx_casted =
        rewriter.create<mlir::arith::IndexCastOp>(loc, itype, index);
    Value bitmask =
        rewriter.create<mlir::arith::ShLIOp>(loc, const_1, idx_casted);

    // take the inverse of bitmask
    Value one_bit = rewriter.create<mlir::arith::ConstantIntOp>(loc, 1, 1);
    Value all_one_mask =
        rewriter.create<mlir::arith::ExtSIOp>(loc, itype, one_bit);
    Value inversed_mask =
        rewriter.create<mlir::arith::XOrIOp>(loc, all_one_mask, bitmask);

    // If val == 1, SetBit should be input OR bitmask (e.g. input || 000010000)
    Value Val1Res = rewriter.create<mlir::arith::OrIOp>(loc, input, bitmask);

    // If val == 0, SetBit should be input AND inversed bitmask
    // (e.g. input && 111101111)
    Value Val0Res =
        rewriter.create<mlir::arith::AndIOp>(loc, input, inversed_mask);
    Value trueRes =
        rewriter.create<arith::SelectOp>(loc, val, Val1Res, Val0Res);

    op->getResult(0).replaceAllUsesWith(trueRes);
  }

  // remove the setBitOps
  std::reverse(setBitOps.begin(), setBitOps.end());
  for (auto op : setBitOps) {
    auto v = op->getResult(0);
    if (v.use_empty()) {
      op->erase();
    }
  }
}

void lowerGetBitOps(func::FuncOp &func) {
  SmallVector<Operation *, 8> getBitOps;
  func.walk([&](Operation *op) {
    if (auto getBitOp = dyn_cast<GetIntBitOp>(op)) {
      getBitOps.push_back(getBitOp);
    }
  });

  for (auto op : getBitOps) {
    auto getBitOp = dyn_cast<GetIntBitOp>(op);
    Value input = getBitOp.getOperand(0);
    Value idx = getBitOp.getOperand(1);
    Location loc = op->getLoc();
    OpBuilder rewriter(op);

    unsigned iwidth = input.getType().getIntOrFloatBitWidth();
    Type itype = rewriter.getIntegerType(iwidth);
    Type i1 = rewriter.getI1Type();
    Value idx_casted =
        rewriter.create<mlir::arith::IndexCastOp>(loc, itype, idx);
    Value shifted =
        rewriter.create<mlir::arith::ShRSIOp>(loc, input, idx_casted);
    Value singleBit = rewriter.create<mlir::arith::TruncIOp>(loc, i1, shifted);

    op->getResult(0).replaceAllUsesWith(singleBit);
  }

  // remove the getBitOps
  std::reverse(getBitOps.begin(), getBitOps.end());
  for (auto op : getBitOps) {
    auto v = op->getResult(0);
    if (v.use_empty()) {
      op->erase();
    }
  }
}

void lowerGetSliceOps(func::FuncOp &func) {
  SmallVector<Operation *, 8> getSliceOps;
  func.walk([&](Operation *op) {
    if (auto getSliceOp = dyn_cast<GetIntSliceOp>(op)) {
      getSliceOps.push_back(getSliceOp);
    }
  });

  for (auto op : getSliceOps) {
    auto getSliceOp = dyn_cast<GetIntSliceOp>(op);
    Value input = getSliceOp.getOperand(0);
    Value hi = getSliceOp.getOperand(1);
    Value lo = getSliceOp.getOperand(2);
    Location loc = op->getLoc();
    OpBuilder rewriter(op);

    // cast low and high index to itype
    unsigned iwidth = input.getType().getIntOrFloatBitWidth();
    Type itype = rewriter.getIntegerType(iwidth);

    Value lo_casted = rewriter.create<mlir::arith::IndexCastOp>(loc, itype, lo);
    Value hi_casted = rewriter.create<mlir::arith::IndexCastOp>(loc, itype, hi);
    Value width = rewriter.create<mlir::arith::ConstantIntOp>(
        loc, input.getType().getIntOrFloatBitWidth() - 1, iwidth);
    Value lshift_width =
        rewriter.create<mlir::arith::SubIOp>(loc, width, hi_casted);

    // We do three shifts to extract the target bit slices
    Value shift1 =
        rewriter.create<mlir::arith::ShLIOp>(loc, input, lshift_width);
    Value shift2 =
        rewriter.create<mlir::arith::ShRUIOp>(loc, shift1, lshift_width);
    Value shift3 =
        rewriter.create<mlir::arith::ShRUIOp>(loc, shift2, lo_casted);
    Type otype = op->getResult(0).getType();
    Value truncated =
        rewriter.create<mlir::arith::TruncIOp>(loc, otype, shift3);

    op->getResult(0).replaceAllUsesWith(truncated);
  }

  // remove the getSliceOps
  std::reverse(getSliceOps.begin(), getSliceOps.end());
  for (auto op : getSliceOps) {
    auto v = op->getResult(0);
    if (v.use_empty()) {
      op->erase();
    }
  }
}

/// Pass entry point
bool applyLowerBitOps(ModuleOp &mod) {
  // First pass: lower bit reverse and set slice
  for (func::FuncOp func : mod.getOps<func::FuncOp>()) {
    lowerBitReverseOps(func);
    lowerSetSliceOps(func);
  }
  // second pass: lower get, set bit, and get slice
  for (func::FuncOp func : mod.getOps<func::FuncOp>()) {
    lowerGetBitOps(func);
    lowerSetBitOps(func);
    lowerGetSliceOps(func);
  }
  return true;
}
} // namespace allo
} // namespace mlir

namespace {
struct AlloLowerBitOpsTransformation
    : public LowerBitOpsBase<AlloLowerBitOpsTransformation> {
  void runOnOperation() override {
    auto mod = getOperation();
    if (!applyLowerBitOps(mod)) {
      return signalPassFailure();
    }
  }
};
} // namespace

namespace mlir {
namespace allo {

std::unique_ptr<OperationPass<ModuleOp>> createLowerBitOpsPass() {
  return std::make_unique<AlloLowerBitOpsTransformation>();
}
} // namespace allo
} // namespace mlir