/*
 * Copyright HeteroCL authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hcl/Conversion/Passes.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/Sequence.h"

using namespace mlir;
using namespace hcl;

namespace {

class CreateLoopHandleOpLowering : public ConversionPattern {
public:
  explicit CreateLoopHandleOpLowering(MLIRContext *context)
      : ConversionPattern(hcl::CreateLoopHandleOp::getOperationName(), 1,
                          context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

class CreateOpHandleOpLowering : public ConversionPattern {
public:
  explicit CreateOpHandleOpLowering(MLIRContext *context)
      : ConversionPattern(hcl::CreateOpHandleOp::getOperationName(), 1,
                          context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

class SetIntBitOpLowering : public ConversionPattern {
public:
  explicit SetIntBitOpLowering(MLIRContext *context)
      : ConversionPattern(hcl::SetIntBitOp::getOperationName(), 3, context) {}
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // SetIntBitOp should be lowered to left shift and bitwise AND/OR
    Value input = operands[0];
    Value index = operands[1];
    Value val = operands[2];
    Location loc = op->getLoc();
    // Cast val to the same with as input
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
    rewriter.replaceOp(op, trueRes);
    return success();
  }
};

class GetIntBitOpLowering : public ConversionPattern {
public:
  explicit GetIntBitOpLowering(MLIRContext *context)
      : ConversionPattern(hcl::GetIntBitOp::getOperationName(), 2, context) {}
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // GetIntBitOp should be lowered to right shift and truncation
    Value input = operands[0];
    Value idx = operands[1];
    Location loc = op->getLoc();
    unsigned iwidth = input.getType().getIntOrFloatBitWidth();
    Type itype = rewriter.getIntegerType(iwidth);
    Type i1 = rewriter.getI1Type();
    Value idx_casted =
        rewriter.create<mlir::arith::IndexCastOp>(loc, itype, idx);
    Value shifted =
        rewriter.create<mlir::arith::ShRSIOp>(loc, input, idx_casted);
    Value singleBit = rewriter.create<mlir::arith::TruncIOp>(loc, i1, shifted);
    op->getResult(0).replaceAllUsesWith(singleBit);
    rewriter.eraseOp(op);
    return success();
  }
};

class GetIntSliceOpLowering : public ConversionPattern {
public:
  explicit GetIntSliceOpLowering(MLIRContext *context)
      : ConversionPattern(hcl::GetIntSliceOp::getOperationName(), 4, context) {}
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = operands[0];
    Value hi = operands[1];
    Value lo = operands[2];
    // cast low and high index to itype
    unsigned iwidth = input.getType().getIntOrFloatBitWidth();
    Type itype = rewriter.getIntegerType(iwidth);
    Location loc = op->getLoc();
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
    rewriter.eraseOp(op);
    return success();
  }
};

/*
class SetIntSliceOpLowering : public ConversionPattern {
public:
  explicit SetIntSliceOpLowering(MLIRContext *context)
      : ConversionPattern(hcl::SetIntSliceOp::getOperationName(), 4, context) {}
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // Three steps to implement setslice:
    // 1. Get higher slice
    // 2. Get lower slice
    // 3. Shift value and use bitwise OR to get result
    Value input = operands[0];
    Value hi = operands[1];
    Value lo = operands[2];
    Value val = operands[3];
    Location loc = op->getLoc();
    // Cast hi, lo to int32, cast val to same dtype as input
    // Note: val's width may be different than (hi-low+1), so
    // we need to clear the peripheral bits.
    unsigned iwidth = input.getType().getIntOrFloatBitWidth();
    Value width =
        rewriter.create<mlir::arith::ConstantIntOp>(loc, iwidth, iwidth);
    Type int_type = rewriter.getIntegerType(iwidth);
    Value lo_casted =
        rewriter.create<mlir::arith::IndexCastOp>(loc, int_type, lo);
    Value hi_casted =
        rewriter.create<mlir::arith::IndexCastOp>(loc, int_type, hi);
    Value const1 =
        rewriter.create<mlir::arith::ConstantIntOp>(loc, 1, int_type);
    Value val_ext =
        rewriter.create<mlir::arith::ExtUIOp>(loc, input.getType(), val);

    // Step 1: get higher slice - shift right, then shift left
    Value hi_shift_width =
        rewriter.create<mlir::arith::AddIOp>(loc, hi_casted, const1);
    Value hi_rshifted =
        rewriter.create<mlir::arith::ShRUIOp>(loc, input, hi_shift_width);
    Value hi_slice =
        rewriter.create<mlir::arith::ShLIOp>(loc, hi_rshifted, hi_shift_width);

    // Step 2: get lower slice - shift left, then shift right
    Value lo_shift_width =
        rewriter.create<mlir::arith::SubIOp>(loc, width, lo_casted);
    Value lo_lshifted =
        rewriter.create<mlir::arith::ShLIOp>(loc, input, lo_shift_width);
    Value lo_slice =
        rewriter.create<mlir::arith::ShRUIOp>(loc, lo_lshifted, lo_shift_width);

    // Step 3: shift left val, and then use OR to "concat" three pieces
    Value val_shifted =
        rewriter.create<mlir::arith::ShLIOp>(loc, val_ext, lo_casted);
    Value peripheral_slices =
        rewriter.create<mlir::arith::OrIOp>(loc, hi_slice, lo_slice);
    Value res = rewriter.create<mlir::arith::OrIOp>(loc, peripheral_slices,
                                                    val_shifted);

    op->getOperand(0).replaceAllUsesWith(res);
    rewriter.eraseOp(op);
    return success();
  }
};
*/

} // namespace

namespace {
struct HCLToLLVMLoweringPass
    : public HCLToLLVMLoweringBase<HCLToLLVMLoweringPass> {
  void runOnOperation() override {
    auto module = getOperation();
    if (!applyHCLToLLVMLoweringPass(module, getContext()))
      signalPassFailure();
  }
};
} // namespace

namespace mlir {
namespace hcl {
bool applyHCLToLLVMLoweringPass(ModuleOp &module, MLIRContext &context) {
  // The first thing to define is the conversion target. This will define the
  // final target for this lowering. For this lowering, we are only targeting
  // the LLVM dialect.
  LLVMConversionTarget target(context);
  target.addLegalOp<ModuleOp>();

  // During this lowering, we will also be lowering the MemRef types, that are
  // currently being operated on, to a representation in LLVM. To perform this
  // conversion we use a TypeConverter as part of the lowering. This converter
  // details how one type maps to another. This is necessary now that we will be
  // doing more complicated lowerings, involving loop region arguments.
  LLVMTypeConverter typeConverter(&context);

  // Now that the conversion target has been defined, we need to provide the
  // patterns used for lowering. At this point of the compilation process, we
  // have a combination of `hcl`, `affine`, and `std` operations. Luckily, there
  // are already exists a set of patterns to transform `affine` and `std`
  // dialects. These patterns lowering in multiple stages, relying on transitive
  // lowerings. Transitive lowering, or A->B->C lowering, is when multiple
  // patterns must be applied to fully transform an illegal operation into a
  // set of legal ones.
  RewritePatternSet patterns(&context);
  populateAffineToStdConversionPatterns(patterns);
  populateSCFToControlFlowConversionPatterns(patterns);

  populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
  arith::populateArithExpandOpsPatterns(patterns);
  arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);

  populateExpandCtlzPattern(patterns);
  populateExpandTanhPattern(patterns);
  populateMathAlgebraicSimplificationPatterns(patterns);
  populateMathPolynomialApproximationPatterns(patterns);
  populateMathToLLVMConversionPatterns(typeConverter, patterns);

  populateFuncToLLVMConversionPatterns(typeConverter, patterns);
  cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
//   populateReconcileUnrealizedCastsPatterns(patterns);

  patterns.add<CreateLoopHandleOpLowering>(&context);
  patterns.add<CreateOpHandleOpLowering>(&context);
  patterns.add<SetIntBitOpLowering>(&context);
  patterns.add<GetIntBitOpLowering>(&context);
  //   patterns.add<SetIntSliceOpLowering>(&context);
  patterns.add<GetIntSliceOpLowering>(&context);

  // We want to completely lower to LLVM, so we use a `FullConversion`. This
  // ensures that only legal operations will remain after the conversion.
  if (failed(applyFullConversion(module, target, std::move(patterns))))
    return false;
  return true;
}
} // namespace hcl
} // namespace mlir

namespace mlir {
namespace hcl {
// void registerHCLToLLVMLoweringPass() {
//   PassRegistration<HCLToLLVMLoweringPass>();
// }

std::unique_ptr<OperationPass<ModuleOp>> createHCLToLLVMLoweringPass() {
  return std::make_unique<HCLToLLVMLoweringPass>();
}
} // namespace hcl
} // namespace mlir
