/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PassDetail.h"
#include "allo/Dialect/AlloDialect.h"
#include "allo/Dialect/AlloOps.h"
#include "allo/Dialect/AlloTypes.h"
#include "allo/Support/Utils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/MLIRContext.h"

using namespace mlir;
using namespace allo;

namespace mlir {
namespace allo {
#define GEN_PASS_DEF_FIXEDTOINTEGER
#include "allo/Conversion/Passes.h.inc"
} // namespace allo
} // namespace mlir

namespace mlir {
namespace allo {

struct FixedTypeInfo {
  size_t width;
  size_t frac;
  bool isSigned;
};

FixedTypeInfo getFixedPointInfo(Type t) {
  FixedTypeInfo info;
  if (FixedType ft = dyn_cast<FixedType>(t)) {
    info.width = ft.getWidth();
    info.frac = ft.getFrac();
    info.isSigned = true;
  } else if (UFixedType uft = dyn_cast<UFixedType>(t)) {
    info.width = uft.getWidth();
    info.frac = uft.getFrac();
    info.isSigned = false;
  } else {
    llvm::errs() << "getFixedPointInfo: not a fixed-point type <" << t << ">\n";
    assert(false);
  }
  return info;
}

Value castIntegerWidth(MLIRContext *ctx, OpBuilder &builder, Location loc,
                       Value v, size_t srcWidth, size_t tgtWidth,
                       bool isSigned) {
  Value result;
  Type newType = IntegerType::get(ctx, tgtWidth);
  if (srcWidth < tgtWidth) {
    // extend bits
    if (isSigned) {
      result = builder.create<arith::ExtSIOp>(loc, newType, v);
    } else {
      result = builder.create<arith::ExtUIOp>(loc, newType, v);
    }
  } else if (srcWidth > tgtWidth) {
    // truncate bits
    result = builder.create<arith::TruncIOp>(loc, newType, v);
  } else {
    result = v;
  }
  return result;
}

Type convertFixedMemRefOrScalarToInt(Type t, MLIRContext *ctx) {
  if (MemRefType memrefType = llvm::dyn_cast<MemRefType>(t)) {
    // if type is memref
    Type et = memrefType.getElementType();
    if (llvm::isa<FixedType, UFixedType>(et)) {
      // if memref element type is fixed-point
      FixedTypeInfo ti = getFixedPointInfo(et);
      Type newElementType = IntegerType::get(ctx, ti.width);
      return memrefType.clone(newElementType);
    } else {
      // if memref element type is not fixed-point
      // make no change
      return t;
    }
  } else {
    // If type is not memref
    if (llvm::isa<FixedType, UFixedType>(t)) {
      // if type is fixed-point
      FixedTypeInfo ti = getFixedPointInfo(t);
      Type newType = IntegerType::get(ctx, ti.width);
      return newType;
    } else {
      // if type is not fixed-point
      // make no change
      return t;
    }
  }
}

void updateFunctionSignature(func::FuncOp &funcOp) {
  FunctionType functionType = funcOp.getFunctionType();
  SmallVector<Type, 4> result_types =
      llvm::to_vector<4>(functionType.getResults());
  SmallVector<Type, 8> arg_types = llvm::to_vector<8>(functionType.getInputs());

  SmallVector<Type, 4> new_result_types;
  SmallVector<Type, 8> new_arg_types;

  for (auto v : llvm::enumerate(result_types)) {
    Type t = v.value();
    Type convertedType =
        convertFixedMemRefOrScalarToInt(t, funcOp.getContext());
    new_result_types.push_back(convertedType);
  }

  for (auto v : llvm::enumerate(arg_types)) {
    Type t = v.value();
    Type convertedType =
        convertFixedMemRefOrScalarToInt(t, funcOp.getContext());
    new_arg_types.push_back(convertedType);
  }

  // Update func::FuncOp's block argument types
  for (Block &block : funcOp.getBlocks()) {
    for (unsigned i = 0; i < block.getNumArguments(); i++) {
      Type argType = block.getArgument(i).getType();
      Type newType =
          convertFixedMemRefOrScalarToInt(argType, funcOp.getContext());
      block.getArgument(i).setType(newType);
    }
  }

  // Update function signature
  FunctionType newFuncType =
      FunctionType::get(funcOp.getContext(), new_arg_types, new_result_types);
  funcOp.setType(newFuncType);
  return;
}

/* Update AffineLoad's result type
After we changed the function arguments, affine loads's argument
memref may change as well, which makes the affine load's result
type different from input memref's element type. This function
updates the result type of affine load operations
*/
void updateAffineLoadStore(func::FuncOp &f) {
  SmallVector<Operation *, 10> loads;
  SmallVector<Operation *, 10> stores;
  f.walk([&](Operation *op) {
    if (auto add_op = dyn_cast<AffineLoadOp>(op)) {
      loads.push_back(op);
    } else if (auto add_op = dyn_cast<AffineStoreOp>(op)) {
      stores.push_back(op);
    }
  });

  for (auto op : loads) {
    for (auto v : llvm::enumerate(op->getResults())) {
      Type newType = llvm::dyn_cast<MemRefType>(op->getOperand(0).getType())
                         .getElementType();
      op->getResult(v.index()).setType(newType);
    }
  }
  for (auto op : stores) {
    Type newType = llvm::dyn_cast<MemRefType>(op->getOperand(1).getType())
                       .getElementType();
    op->getOperand(0).setType(newType);
  }
}

void updateSelectOp(arith::SelectOp &selectOp) {
  // update the result of select op
  // from fixed-point type to integer type
  Type resType = selectOp.getResult().getType();
  if (llvm::isa<FixedType, UFixedType>(resType)) {
    int bitwidth = llvm::isa<FixedType>(resType)
                       ? llvm::dyn_cast<FixedType>(resType).getWidth()
                       : llvm::dyn_cast<UFixedType>(resType).getWidth();
    Type newType = IntegerType::get(selectOp.getContext(), bitwidth);
    selectOp.getResult().setType(newType);
  }
}

/* Update allo.print (PrintOp) operations.
 * Create a float64 memref to store the real value
 * of allo.print's operand memref
 */
void lowerPrintMemRefOp(func::FuncOp &funcOp) {
  SmallVector<Operation *, 4> printOps;
  funcOp.walk([&](Operation *op) {
    if (auto new_op = dyn_cast<PrintMemRefOp>(op)) {
      // Only lower fixed-point prints
      MemRefType memRefType =
          llvm::dyn_cast<MemRefType>(new_op->getOperand(0).getType());
      Type elemType = memRefType.getElementType();
      if (llvm::isa<FixedType, UFixedType>(elemType))
        printOps.push_back(op);
    }
  });
  for (auto *printOp : printOps) {
    OpBuilder builder(printOp);
    Type F64 = builder.getF64Type();
    Location loc = printOp->getLoc();
    Value oldMemRef = printOp->getOperand(0);
    MemRefType oldMemRefType = llvm::dyn_cast<MemRefType>(oldMemRef.getType());
    Type oldType = oldMemRefType.getElementType();
    MemRefType newMemRefType =
        llvm::dyn_cast<MemRefType>(oldMemRefType.clone(F64));
    Value newMemRef = builder.create<memref::AllocOp>(loc, newMemRefType);
    SmallVector<int64_t, 4> lbs(oldMemRefType.getRank(), 0);
    SmallVector<int64_t, 4> steps(oldMemRefType.getRank(), 1);
    buildAffineLoopNest(
        builder, loc, lbs, oldMemRefType.getShape(), steps,
        [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
          Value v = nestedBuilder.create<AffineLoadOp>(loc, oldMemRef, ivs);
          Value casted;
          size_t frac;
          if (llvm::isa<FixedType>(oldType)) {
            casted = nestedBuilder.create<arith::SIToFPOp>(loc, F64, v);
            frac = llvm::dyn_cast<FixedType>(oldType).getFrac();
          } else {
            casted = nestedBuilder.create<arith::UIToFPOp>(loc, F64, v);
            frac = llvm::dyn_cast<UFixedType>(oldType).getFrac();
          }
          Value const_frac = nestedBuilder.create<mlir::arith::ConstantOp>(
              loc, F64, nestedBuilder.getFloatAttr(F64, std::pow(2, frac)));
          Value realV = nestedBuilder.create<mlir::arith::DivFOp>(
              loc, F64, casted, const_frac);
          nestedBuilder.create<AffineStoreOp>(loc, realV, newMemRef, ivs);
        });
    printOp->setOperand(0, newMemRef);
  }
}

void lowerPrintOp(func::FuncOp &funcOp) {
  SmallVector<Operation *, 4> printOps;
  funcOp.walk([&](Operation *op) {
    if (auto new_op = dyn_cast<PrintOp>(op)) {
      // Only lower fixed-point prints
      for (auto operand : new_op->getOperands()) {
        if (llvm::isa<FixedType, UFixedType>(operand.getType())) {
          printOps.push_back(op);
          break;
        }
      }
    }
  });

  for (auto *printOp : printOps) {
    for (auto opr : llvm::enumerate(printOp->getOperands())) {
      if (llvm::isa<FixedType, UFixedType>(opr.value().getType())) {
        OpBuilder builder(printOp);
        Value oldValue = opr.value();
        bool is_unsigned = llvm::isa<UFixedType>(opr.value().getType());
        Value newValue = castToF64(builder, oldValue, is_unsigned);
        printOp->setOperand(opr.index(), newValue);
      }
    }
  }
}

// Fixed-point memref allocation op to integer memref
void updateAlloc(func::FuncOp &f) {
  SmallVector<Operation *, 10> allocOps;
  f.walk([&](Operation *op) {
    if (auto alloc_op = dyn_cast<memref::AllocOp>(op)) {
      allocOps.push_back(op);
    }
  });

  for (auto op : allocOps) {
    auto allocOp = dyn_cast<memref::AllocOp>(op);
    Type newType =
        convertFixedMemRefOrScalarToInt(allocOp.getType(), f.getContext());
    op->getResult(0).setType(newType);
  }
}

void updateSCFIfOp(mlir::scf::IfOp &op) {
  for (auto res : op.getResults()) {
    if (llvm::isa<FixedType>(res.getType())) {
      res.setType(IntegerType::get(
          res.getContext(),
          llvm::dyn_cast<FixedType>(res.getType()).getWidth()));
    } else if (llvm::isa<UFixedType>(res.getType())) {
      res.setType(IntegerType::get(
          res.getContext(),
          llvm::dyn_cast<UFixedType>(res.getType()).getWidth()));
    } else if (auto memRefType = llvm::dyn_cast<MemRefType>(res.getType())) {
      Type eleTyp = memRefType.getElementType();
      if (llvm::isa<FixedType>(eleTyp)) {
        eleTyp = IntegerType::get(res.getContext(),
                                  llvm::dyn_cast<FixedType>(eleTyp).getWidth());
      } else if (llvm::isa<UFixedType>(eleTyp)) {
        eleTyp = IntegerType::get(
            res.getContext(), llvm::dyn_cast<UFixedType>(eleTyp).getWidth());
      }
      res.setType(memRefType.clone(eleTyp));
    }
  }
}

// Lower AddFixedOp to AddIOp
void lowerFixedAdd(AddFixedOp &op) {
  OpBuilder rewriter(op);
  Type t = op->getOperand(0).getType();
  FixedTypeInfo ti = getFixedPointInfo(t);
  auto lhs = op->getOperand(0);
  auto rhs = op->getOperand(1);
  Type newType = IntegerType::get(op.getContext(), ti.width);
  arith::AddIOp newOp =
      rewriter.create<arith::AddIOp>(op->getLoc(), newType, lhs, rhs);
  op->replaceAllUsesWith(newOp);
}

// Lower FixedSubOp to SubIOp
void lowerFixedSub(SubFixedOp &op) {
  OpBuilder rewriter(op);
  Type t = op->getOperand(0).getType();
  FixedTypeInfo ti = getFixedPointInfo(t);
  auto lhs = op->getOperand(0);
  auto rhs = op->getOperand(1);
  Type newType = IntegerType::get(op.getContext(), ti.width);
  arith::SubIOp newOp =
      rewriter.create<arith::SubIOp>(op->getLoc(), newType, lhs, rhs);
  op->replaceAllUsesWith(newOp);
}

// Lower MulFixedop to MulIOp
void lowerFixedMul(MulFixedOp &op) {
  Type t = op->getOperand(0).getType();
  FixedTypeInfo ti = getFixedPointInfo(t);
  OpBuilder rewriter(op);
  Value lhs =
      castIntegerWidth(op->getContext(), rewriter, op->getLoc(),
                       op->getOperand(0), ti.width, ti.width * 2, ti.isSigned);
  Value rhs =
      castIntegerWidth(op->getContext(), rewriter, op->getLoc(),
                       op->getOperand(1), ti.width, ti.width * 2, ti.isSigned);
  IntegerType intTy = IntegerType::get(op->getContext(), ti.width * 2);
  IntegerType truncTy = IntegerType::get(op->getContext(), ti.width);
  arith::MulIOp newOp =
      rewriter.create<arith::MulIOp>(op->getLoc(), intTy, lhs, rhs);

  // lhs<width, frac> * rhs<width, frac> -> res<width, 2*frac>
  // Therefore, we need to right shift the result for frac bit
  // Right shift needs to consider signed/unsigned
  auto fracAttr = rewriter.getIntegerAttr(intTy, ti.frac);
  auto fracCstOp =
      rewriter.create<arith::ConstantOp>(op->getLoc(), intTy, fracAttr);

  if (ti.isSigned) {
    // use signed right shift
    arith::ShRSIOp res =
        rewriter.create<arith::ShRSIOp>(op->getLoc(), newOp, fracCstOp);
    auto truncated =
        rewriter.create<arith::TruncIOp>(op->getLoc(), truncTy, res);
    op->replaceAllUsesWith(truncated);
  } else {
    // use unsigned right shift
    arith::ShRUIOp res =
        rewriter.create<arith::ShRUIOp>(op->getLoc(), newOp, fracCstOp);
    auto truncated =
        rewriter.create<arith::TruncIOp>(op->getLoc(), truncTy, res);
    op->replaceAllUsesWith(truncated);
  }
}

// Lower FixedDivOp to DivSIOp/DivUIOp
void lowerFixedDiv(DivFixedOp &op) {
  Type t = op->getOperand(0).getType();
  FixedTypeInfo ti = getFixedPointInfo(t);
  OpBuilder rewriter(op);
  Value lhs =
      castIntegerWidth(op->getContext(), rewriter, op->getLoc(),
                       op->getOperand(0), ti.width, ti.width * 2, ti.isSigned);
  Value rhs =
      castIntegerWidth(op->getContext(), rewriter, op->getLoc(),
                       op->getOperand(1), ti.width, ti.width * 2, ti.isSigned);
  // lhs<width, frac> / rhs<width, frac> -> res<width, 0>
  // Therefore, we need to left shift the lhs for frac bit
  // lhs<width, 2 * frac> / rhs<width, frac> -> res<width, frac>
  IntegerType intTy = IntegerType::get(op->getContext(), ti.width * 2);
  IntegerType truncTy = IntegerType::get(op->getContext(), ti.width);
  auto fracAttr = rewriter.getIntegerAttr(intTy, ti.frac);
  auto fracCstOp =
      rewriter.create<arith::ConstantOp>(op->getLoc(), intTy, fracAttr);
  arith::ShLIOp lhs_shifted =
      rewriter.create<arith::ShLIOp>(op->getLoc(), lhs, fracCstOp);
  if (ti.isSigned) { // signed fixed
    arith::DivSIOp res =
        rewriter.create<arith::DivSIOp>(op->getLoc(), lhs_shifted, rhs);
    auto truncated =
        rewriter.create<arith::TruncIOp>(op->getLoc(), truncTy, res);
    op->replaceAllUsesWith(truncated);
  } else { // ufixed
    arith::DivUIOp res =
        rewriter.create<arith::DivUIOp>(op->getLoc(), lhs_shifted, rhs);
    auto truncated =
        rewriter.create<arith::TruncIOp>(op->getLoc(), truncTy, res);
    op->replaceAllUsesWith(truncated);
  }
}

// Lower ShLFixedOp to ShLIOp
// https://docs.amd.com/r/en-US/ug1399-vitis-hls/Class-Methods-Operators-and-Data-Members
void lowerFixedShL(ShLFixedOp &op) {
  OpBuilder rewriter(op);
  Type t = op->getOperand(0).getType();
  FixedTypeInfo ti = getFixedPointInfo(t);
  auto lhs = op->getOperand(0);
  int sh_width =
      llvm::dyn_cast<IntegerType>(op->getOperand(1).getType()).getWidth();
  Value rhs =
      castIntegerWidth(op->getContext(), rewriter, op->getLoc(),
                       op->getOperand(1), sh_width, ti.width, ti.isSigned);
  Type newType = IntegerType::get(op.getContext(), ti.width);
  arith::ShLIOp newOp =
      rewriter.create<arith::ShLIOp>(op->getLoc(), newType, lhs, rhs);
  op->replaceAllUsesWith(newOp);
}

// Lower ShRFixedOp to ShRIOp
void lowerFixedShR(ShRFixedOp &op) {
  OpBuilder rewriter(op);
  Type t = op->getOperand(0).getType();
  FixedTypeInfo ti = getFixedPointInfo(t);
  auto lhs = op->getOperand(0);
  int sh_width =
      llvm::dyn_cast<IntegerType>(op->getOperand(1).getType()).getWidth();
  Value rhs =
      castIntegerWidth(op->getContext(), rewriter, op->getLoc(),
                       op->getOperand(1), sh_width, ti.width, ti.isSigned);
  Type newType = IntegerType::get(op.getContext(), ti.width);
  arith::ShRSIOp newOp =
      rewriter.create<arith::ShRSIOp>(op->getLoc(), newType, lhs, rhs);
  op->replaceAllUsesWith(newOp);
}

// Lower CmpFixedOp to CmpIOp
void lowerFixedCmp(CmpFixedOp &op) {
  OpBuilder rewriter(op);
  auto lhs = op->getOperand(0);
  auto rhs = op->getOperand(1);
  auto prednum = op.getPredicate();
  auto loc = op->getLoc();
  arith::CmpIOp newOp;
  switch (prednum) {
  case allo::CmpFixedPredicate::eq:
    newOp =
        rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, lhs, rhs);
    break;
  case allo::CmpFixedPredicate::ne:
    newOp =
        rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, lhs, rhs);
    break;
  case allo::CmpFixedPredicate::slt:
    newOp = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, lhs,
                                           rhs);
    break;
  case allo::CmpFixedPredicate::sle:
    newOp = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sle, lhs,
                                           rhs);
    break;
  case allo::CmpFixedPredicate::sgt:
    newOp = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, lhs,
                                           rhs);
    break;
  case allo::CmpFixedPredicate::sge:
    newOp = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, lhs,
                                           rhs);
    break;
  case allo::CmpFixedPredicate::ult:
    newOp = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, lhs,
                                           rhs);
    break;
  case allo::CmpFixedPredicate::ule:
    newOp = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ule, lhs,
                                           rhs);
    break;
  case allo::CmpFixedPredicate::ugt:
    newOp = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, lhs,
                                           rhs);
    break;
  case allo::CmpFixedPredicate::uge:
    newOp = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::uge, lhs,
                                           rhs);
    break;
  default:
    llvm::errs() << "unknown predicate code in CmpFixedOp\n";
  }

  op->replaceAllUsesWith(newOp);
}

// Lower MinFixedOp to MinSIOp or MinUIOp
void lowerFixedMin(MinFixedOp &op) {
  Type t = op->getOperand(0).getType();
  FixedTypeInfo ti = getFixedPointInfo(t);
  OpBuilder rewriter(op);
  auto lhs = op->getOperand(0);
  auto rhs = op->getOperand(1);
  if (ti.isSigned) {
    // use signed integer min
    auto res = rewriter.create<arith::MinSIOp>(op->getLoc(), lhs, rhs);
    op->replaceAllUsesWith(res);
  } else {
    // use unsigned integer min
    auto res = rewriter.create<arith::MinUIOp>(op->getLoc(), lhs, rhs);
    op->replaceAllUsesWith(res);
  }
}

// Lower MaxFixedOp to MaxSIOp or MaxUIOp
void lowerFixedMax(MaxFixedOp &op) {
  Type t = op->getOperand(0).getType();
  FixedTypeInfo ti = getFixedPointInfo(t);
  OpBuilder rewriter(op);
  auto lhs = op->getOperand(0);
  auto rhs = op->getOperand(1);
  if (ti.isSigned) {
    // use signed integer max
    auto res = rewriter.create<arith::MaxSIOp>(op->getLoc(), lhs, rhs);
    op->replaceAllUsesWith(res);
  } else {
    // use unsigned integer max
    auto res = rewriter.create<arith::MaxUIOp>(op->getLoc(), lhs, rhs);
    op->replaceAllUsesWith(res);
  }
}

// Build a memref.get_global operation that points to an I64 global memref
// The assumption is that all fixed-point encoding's global memrefs are of
// type I64.
void lowerGetGlobalFixedOp(GetGlobalFixedOp &op) {
  // TODO(Niansong): truncate the global memref to the width of the fixed-point
  OpBuilder rewriter(op);
  auto loc = op.getLoc();
  MemRefType oldType = llvm::dyn_cast<MemRefType>(op->getResult(0).getType());
  Type oldElementType = oldType.getElementType();
  FixedTypeInfo ti = getFixedPointInfo(oldElementType);
  auto memRefType = oldType.clone(IntegerType::get(op.getContext(), 64));
  auto symbolName = op.getName();
  auto res = rewriter.create<memref::GetGlobalOp>(loc, memRefType, symbolName);
  // Truncate or Extend I64 memref to the width of the fixed-point
  auto castedMemRefType = llvm::dyn_cast<MemRefType>(
      oldType.clone(IntegerType::get(op.getContext(), ti.width)));
  auto castedMemRef = rewriter.create<memref::AllocOp>(loc, castedMemRefType);
  SmallVector<int64_t, 4> lbs(oldType.getRank(), 0);
  SmallVector<int64_t, 4> steps(oldType.getRank(), 1);
  buildAffineLoopNest(
      rewriter, loc, lbs, oldType.getShape(), steps,
      [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
        auto v = nestedBuilder.create<AffineLoadOp>(loc, res, ivs);
        Value casted = castIntegerWidth(op.getContext(), nestedBuilder, loc, v,
                                        64, ti.width, ti.isSigned);
        nestedBuilder.create<AffineStoreOp>(loc, casted, castedMemRef, ivs);
      });

  op->replaceAllUsesWith(castedMemRef);
  // update affine.load operations res type to be consistent with castedMemRef's
  // element type
  for (auto &use : castedMemRef.getResult().getUses()) {
    if (auto loadOp = dyn_cast<AffineLoadOp>(use.getOwner())) {
      for (auto v : llvm::enumerate(loadOp->getResults())) {
        Type newType =
            llvm::dyn_cast<MemRefType>(loadOp->getOperand(0).getType())
                .getElementType();
        loadOp->getResult(v.index()).setType(newType);
      }
    }
  }
}

void lowerFixedToFloat(FixedToFloatOp &op) {
  Type t = op.getOperand().getType();
  FixedTypeInfo ti = getFixedPointInfo(t);
  OpBuilder rewriter(op);
  auto loc = op.getLoc();
  auto src = op.getOperand();
  auto dst = op.getResult();
  auto dstTy = llvm::dyn_cast<FloatType>(dst.getType());
  auto frac = rewriter.create<arith::ConstantOp>(
      loc, dstTy, rewriter.getFloatAttr(dstTy, std::pow(2, ti.frac)));
  if (ti.isSigned) {
    auto res = rewriter.create<arith::SIToFPOp>(loc, dstTy, src);
    auto real = rewriter.create<arith::DivFOp>(loc, dstTy, res, frac);
    op->replaceAllUsesWith(real);
  } else {
    auto res = rewriter.create<arith::UIToFPOp>(loc, dstTy, src);
    auto real = rewriter.create<arith::DivFOp>(loc, dstTy, res, frac);
    op->replaceAllUsesWith(real);
  }
}

void lowerFloatToFixed(FloatToFixedOp &op) {
  OpBuilder rewriter(op);
  auto loc = op.getLoc();
  auto src = op.getOperand();
  Type t = op.getResult().getType();
  FixedTypeInfo ti = getFixedPointInfo(t);
  auto FType = llvm::dyn_cast<FloatType>(src.getType());
  auto frac = rewriter.create<arith::ConstantOp>(
      loc, FType, rewriter.getFloatAttr(FType, std::pow(2, ti.frac)));
  auto dstType = IntegerType::get(op.getContext(), ti.width);
  auto FEncoding = rewriter.create<arith::MulFOp>(loc, FType, src, frac);
  if (ti.isSigned) {
    auto IEncoding = rewriter.create<arith::FPToSIOp>(loc, dstType, FEncoding);
    op->replaceAllUsesWith(IEncoding);
  } else {
    auto IEncoding = rewriter.create<arith::FPToUIOp>(loc, dstType, FEncoding);
    op->replaceAllUsesWith(IEncoding);
  }
}

void lowerFixedToInt(FixedToIntOp &op) {
  OpBuilder rewriter(op);
  auto loc = op.getLoc();
  auto src = op.getOperand();
  auto dst = op.getResult();
  Type t = src.getType();
  FixedTypeInfo ti = getFixedPointInfo(t);
  auto src_width = ti.width;
  auto src_frac = ti.frac;
  auto dstType = llvm::dyn_cast<IntegerType>(dst.getType());
  auto srcType = IntegerType::get(op.getContext(), src_width);
  size_t dst_width = dstType.getWidth();
  auto frac = rewriter.create<arith::ConstantOp>(
      loc, srcType, rewriter.getIntegerAttr(srcType, src_frac));
  if (ti.isSigned) {
    auto rshifted = rewriter.create<arith::ShRSIOp>(loc, srcType, src, frac);
    if (dst_width > src_width) {
      auto res = rewriter.create<arith::ExtSIOp>(loc, dstType, rshifted);
      op->replaceAllUsesWith(res);
    } else if (dst_width < src_width) {
      auto res = rewriter.create<arith::TruncIOp>(loc, dstType, rshifted);
      op->replaceAllUsesWith(res);
    } else {
      op->replaceAllUsesWith(rshifted);
    }
  } else {
    auto rshifted = rewriter.create<arith::ShRUIOp>(loc, srcType, src, frac);
    if (dst_width > src_width) {
      auto res = rewriter.create<arith::ExtUIOp>(loc, dstType, rshifted);
      op->replaceAllUsesWith(res);
    } else if (dst_width < src_width) {
      auto res = rewriter.create<arith::TruncIOp>(loc, dstType, rshifted);
      op->replaceAllUsesWith(res);
    } else {
      op->replaceAllUsesWith(rshifted);
    }
  }
}

void lowerIntToFixed(IntToFixedOp &op) {
  OpBuilder rewriter(op);
  auto loc = op.getLoc();
  auto src = op.getOperand();
  auto dst = op.getResult();
  Type t = dst.getType();
  FixedTypeInfo ti = getFixedPointInfo(t);
  auto src_width = llvm::dyn_cast<IntegerType>(src.getType()).getWidth();
  auto dst_width = ti.width;
  auto dst_frac = ti.frac;
  auto dstType = IntegerType::get(op.getContext(), dst_width);
  auto frac = rewriter.create<arith::ConstantOp>(
      loc, dstType, rewriter.getIntegerAttr(dstType, dst_frac));

  Value bitAdjusted = castIntegerWidth(op->getContext(), rewriter, loc, src,
                                       src_width, dst_width, ti.isSigned);
  auto lshifted =
      rewriter.create<arith::ShLIOp>(loc, dstType, bitAdjusted, frac);
  op->replaceAllUsesWith(lshifted);
}

void updateCallOp(func::CallOp &op) {
  // get the callee function signature type
  FunctionType callee_type = op.getCalleeType();
  auto callee = op.getCallee();
  assert(callee != "top" && "updateCallOp: assume callee is not top");
  SmallVector<Type, 4> result_types =
      llvm::to_vector<4>(callee_type.getResults());
  llvm::SmallVector<Type, 4> new_result_types;
  for (Type t : result_types) {
    Type new_type = convertFixedMemRefOrScalarToInt(t, op.getContext());
    new_result_types.push_back(new_type);
  }
  // set call op result types to new_result_types
  for (auto v : llvm::enumerate(new_result_types)) {
    op.getResult(v.index()).setType(v.value());
  }
}

// src and dst is guaranteed to be of different fixed types.
// src: src_width, src_frac
// dst: dst_width, dst_frac
// case 1: src_width > dst_width, src_frac > dst_frac
// case 2: src_width > dst_width, src_frac < dst_frac
// case 3: src_width < dst_width, src_frac > dst_frac
// case 4: src_width < dst_width, src_frac < dst_frac
// src_base * 2^(-src_frac) = dst_base * 2^(-dst_frac)
// ==> dst_base = src_base * 2^(dst_frac - src_frac)
void lowerFixedToFixed(FixedToFixedOp &op) {
  // Step 1: match bitwidth to max(src_width, dst_width)
  // Step 2: shift src_base to get dst_base
  //    - if dst_frac > src_frac, left shift (dst_frac - src_frac)
  //    - if dst_frac < src_frac, right shift (src_frac - dst_frac)
  // Step 3 (optional): truncate dst_base
  OpBuilder rewriter(op);
  auto loc = op.getLoc();
  auto src = op.getOperand();
  auto dst = op.getResult();
  FixedTypeInfo src_ti = getFixedPointInfo(src.getType());
  FixedTypeInfo dst_ti = getFixedPointInfo(dst.getType());
  size_t src_width = src_ti.width;
  size_t src_frac = src_ti.frac;
  size_t dst_width = dst_ti.width;
  size_t dst_frac = dst_ti.frac;
  bool isSignedSrc = src_ti.isSigned;
  auto srcType = IntegerType::get(op.getContext(), src_width);
  auto dstType = IntegerType::get(op.getContext(), dst_width);

  // Step1: match bitwidth to max(src_width, dst_width)
  bool truncate_dst = false;
  bool match_to_dst = false;
  Value matched_src;
  if (dst_width > src_width) {
    // if (dst_width > src_width), no need to truncate dst_base at step3
    truncate_dst = false;
    match_to_dst = true;
    // extend src_base to dst_width
    if (isSignedSrc) {
      matched_src = rewriter.create<arith::ExtSIOp>(loc, dstType, src);
    } else {
      matched_src = rewriter.create<arith::ExtUIOp>(loc, dstType, src);
    }
  } else if (dst_width == src_width) {
    truncate_dst = false;
    match_to_dst = false;
    matched_src = src;
  } else {
    // if (dst_width < src_width), truncate dst_base at step3
    truncate_dst = true;
    match_to_dst = false;
    matched_src = src;
  }

  // Step2: shift src_base to get dst_base
  Value shifted_src;
  if (dst_frac > src_frac) {
    // if (dst_frac > src_frac), left shift (dst_frac - src_frac)
    Type shiftType = match_to_dst ? dstType : srcType;
    auto frac = rewriter.create<arith::ConstantOp>(
        loc, shiftType,
        rewriter.getIntegerAttr(shiftType, dst_frac - src_frac));
    shifted_src =
        rewriter.create<arith::ShLIOp>(loc, shiftType, matched_src, frac);
  } else if (dst_frac < src_frac) {
    // if (dst_frac < src_frac), right shift (src_frac - dst_frac)
    Type shiftType = match_to_dst ? dstType : srcType;
    auto frac = rewriter.create<arith::ConstantOp>(
        loc, shiftType,
        rewriter.getIntegerAttr(shiftType, src_frac - dst_frac));
    if (isSignedSrc) {
      shifted_src =
          rewriter.create<arith::ShRSIOp>(loc, shiftType, matched_src, frac);
    } else {
      shifted_src =
          rewriter.create<arith::ShRUIOp>(loc, shiftType, matched_src, frac);
    }
  } else {
    shifted_src = matched_src;
  }

  // debug output
  // llvm::outs() << shifted_src << "\n";

  // Step3 (optional): truncate dst_base
  if (truncate_dst) {
    auto res = rewriter.create<arith::TruncIOp>(loc, dstType, shifted_src);
    op->replaceAllUsesWith(res);
  } else {
    op->getResult(0).replaceAllUsesWith(shifted_src);
  }
}

void validateLoweredFunc(func::FuncOp &func) {
  // check if result types and input types are not fixed or ufixed
  FunctionType functionType = func.getFunctionType();
  SmallVector<Type, 4> result_types =
      llvm::to_vector<4>(functionType.getResults());
  SmallVector<Type, 8> arg_types = llvm::to_vector<8>(functionType.getInputs());
  for (auto result_type : result_types) {
    if (llvm::isa<FixedType>(result_type) ||
        llvm::isa<UFixedType>(result_type)) {
      func.emitError(
          "FuncOp: " + func.getName().str() +
          " has fixed-point type result type: " +
          " which means it is not lowered by FixedPointToInteger pass\n");
    }
  }
  for (auto arg_type : arg_types) {
    if (llvm::isa<FixedType>(arg_type) || llvm::isa<UFixedType>(arg_type)) {
      func.emitError(
          "FuncOp: " + func.getName().str() +
          " has fixed-point type arg type: " +
          " which means it is not lowered by FixedPointToInteger pass\n");
    }
  }

  // check if all operations are lowered
  for (auto &block : func.getBody().getBlocks()) {
    for (auto &op : block.getOperations()) {
      // check the result type and arg types of op
      if (op.getNumResults() > 0) {
        for (auto result : op.getResults()) {
          if (llvm::isa<FixedType>(result.getType()) ||
              llvm::isa<UFixedType>(result.getType())) {
            op.emitError(
                "FuncOp: " + func.getName().str() +
                " has op: " + std::string(op.getName().getStringRef()) +
                " with fixed-point result type" +
                " which means it is not lowered by FixedPointToInteger pass\n");
            llvm::errs() << "op that failed validation: " << op << "\n";
          }
        }
      }
      // check the arg types of op
      for (auto arg : op.getOperands()) {
        if (llvm::isa<FixedType>(arg.getType()) ||
            llvm::isa<UFixedType>(arg.getType())) {
          op.emitError(
              "FuncOp: " + func.getName().str() +
              " has op: " + std::string(op.getName().getStringRef()) +
              " with fixed-point arg type" +
              " which means it is not lowered by FixedPointToInteger pass\n");
          llvm::errs() << "op that failed validation: " << op << "\n";
        }
      }
    }
  }
}

/// Visitors to recursively update all operations
void visitOperation(Operation &op);
void visitRegion(Region &region);
void visitBlock(Block &block);

void visitOperation(Operation &op) {
  if (auto new_op = dyn_cast<AddFixedOp>(op)) {
    lowerFixedAdd(new_op);
  } else if (auto new_op = dyn_cast<SubFixedOp>(op)) {
    lowerFixedSub(new_op);
  } else if (auto new_op = dyn_cast<MulFixedOp>(op)) {
    lowerFixedMul(new_op);
  } else if (auto new_op = dyn_cast<DivFixedOp>(op)) {
    lowerFixedDiv(new_op);
  } else if (auto new_op = dyn_cast<CmpFixedOp>(op)) {
    lowerFixedCmp(new_op);
  } else if (auto new_op = dyn_cast<ShLFixedOp>(op)) {
    lowerFixedShL(new_op);
  } else if (auto new_op = dyn_cast<ShRFixedOp>(op)) {
    lowerFixedShR(new_op);
  } else if (auto new_op = dyn_cast<MinFixedOp>(op)) {
    lowerFixedMin(new_op);
  } else if (auto new_op = dyn_cast<MaxFixedOp>(op)) {
    lowerFixedMax(new_op);
  } else if (auto new_op = dyn_cast<GetGlobalFixedOp>(op)) {
    lowerGetGlobalFixedOp(new_op);
  } else if (auto new_op = dyn_cast<FixedToFloatOp>(op)) {
    lowerFixedToFloat(new_op);
  } else if (auto new_op = dyn_cast<FloatToFixedOp>(op)) {
    lowerFloatToFixed(new_op);
  } else if (auto new_op = dyn_cast<FixedToIntOp>(op)) {
    lowerFixedToInt(new_op);
  } else if (auto new_op = dyn_cast<IntToFixedOp>(op)) {
    lowerIntToFixed(new_op);
  } else if (auto new_op = dyn_cast<FixedToFixedOp>(op)) {
    lowerFixedToFixed(new_op);
  } else if (auto new_op = dyn_cast<scf::IfOp>(op)) {
    updateSCFIfOp(new_op);
  } else if (auto new_op = dyn_cast<func::CallOp>(op)) {
    updateCallOp(new_op);
  } else if (auto new_op = dyn_cast<arith::SelectOp>(op)) {
    updateSelectOp(new_op);
  }

  for (auto &region : op.getRegions()) {
    visitRegion(region);
  }
}

void visitBlock(Block &block) {
  SmallVector<Operation *, 10> opToRemove;
  for (auto it = block.rbegin(); it != block.rend(); ++it) {
    Operation &op = *it;
    visitOperation(op);
    if (llvm::isa<AddFixedOp, SubFixedOp, MulFixedOp, DivFixedOp, CmpFixedOp,
                  ShLFixedOp, ShRFixedOp, MinFixedOp, MaxFixedOp, IntToFixedOp,
                  FixedToIntOp, FloatToFixedOp, FixedToFloatOp, FixedToFixedOp,
                  GetGlobalFixedOp>(op)) {
      opToRemove.push_back(&op);
    }
  }

  // Remove fixed-point operations after the block
  // is visited.
  std::reverse(opToRemove.begin(), opToRemove.end());
  for (Operation *op : opToRemove) {
    op->erase();
  }
}

void visitRegion(Region &region) {
  for (auto &block : region.getBlocks()) {
    visitBlock(block);
  }
}

/// Pass entry point
bool applyFixedPointToInteger(ModuleOp &mod) {

  for (func::FuncOp func : mod.getOps<func::FuncOp>()) {
    lowerPrintMemRefOp(func);
    lowerPrintOp(func);
    // lower arith, scf, conversion ops
    visitRegion(func.getBody());
    updateFunctionSignature(func);
    updateAlloc(func);
    updateAffineLoadStore(func);
    validateLoweredFunc(func);
  }

  // llvm::outs() << mod << "\n";

  return true;
}
} // namespace allo
} // namespace mlir

namespace {

struct AlloFixedToIntegerTransformation
    : public mlir::allo::impl::FixedToIntegerBase<
          AlloFixedToIntegerTransformation> {

  void runOnOperation() override {
    auto mod = getOperation();
    if (!applyFixedPointToInteger(mod))
      return signalPassFailure();
  }
};

} // namespace

namespace mlir {
namespace allo {

// Create A Fixed-Point to Integer Pass
std::unique_ptr<OperationPass<ModuleOp>> createFixedPointToIntegerPass() {
  return std::make_unique<AlloFixedToIntegerTransformation>();
}

} // namespace allo
} // namespace mlir