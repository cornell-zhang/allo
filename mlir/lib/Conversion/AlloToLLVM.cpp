/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PassDetail.h"

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
using namespace allo;

namespace mlir {
namespace allo {
#define GEN_PASS_DEF_ALLOTOLLVMLOWERING
#include "allo/Conversion/Passes.h.inc"
} // namespace allo
} // namespace mlir

namespace {

class CreateLoopHandleOpLowering : public ConversionPattern {
public:
  explicit CreateLoopHandleOpLowering(MLIRContext *context)
      : ConversionPattern(allo::CreateLoopHandleOp::getOperationName(), 1,
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
      : ConversionPattern(allo::CreateOpHandleOp::getOperationName(), 1,
                          context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

namespace {
struct AlloToLLVMLoweringPass
    : public mlir::allo::impl::AlloToLLVMLoweringBase<AlloToLLVMLoweringPass> {
  void runOnOperation() override {
    auto module = getOperation();
    if (!applyAlloToLLVMLoweringPass(module, getContext()))
      signalPassFailure();
  }
};
} // namespace

namespace mlir {
namespace allo {
bool applyAlloToLLVMLoweringPass(ModuleOp &module, MLIRContext &context) {
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
  // have a combination of `allo`, `affine`, and `std` operations. Luckily,
  // there are already exists a set of patterns to transform `affine` and `std`
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

  populatePolynomialApproximateTanhPattern(patterns);
  populateMathAlgebraicSimplificationPatterns(patterns);
  populateMathPolynomialApproximationPatterns(patterns);
  populateMathToLLVMConversionPatterns(typeConverter, patterns);

  populateFuncToLLVMConversionPatterns(typeConverter, patterns);
  cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
  //   populateReconcileUnrealizedCastsPatterns(patterns);

  patterns.add<CreateLoopHandleOpLowering>(&context);
  patterns.add<CreateOpHandleOpLowering>(&context);

  // We want to completely lower to LLVM, so we use a `FullConversion`. This
  // ensures that only legal operations will remain after the conversion.
  if (failed(applyFullConversion(module, target, std::move(patterns))))
    return false;
  return true;
}
} // namespace allo
} // namespace mlir

namespace mlir {
namespace allo {
// void registerAlloToLLVMLoweringPass() {
//   PassRegistration<AlloToLLVMLoweringPass>();
// }

std::unique_ptr<OperationPass<ModuleOp>> createAlloToLLVMLoweringPass() {
  return std::make_unique<AlloToLLVMLoweringPass>();
}
} // namespace allo
} // namespace mlir
