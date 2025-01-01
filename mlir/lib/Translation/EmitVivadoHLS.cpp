/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 * Modification: ScaleHLS
 * https://github.com/hanchenye/scalehls
 */

#include "allo/Translation/EmitVivadoHLS.h"
#include "allo/Dialect/Visitor.h"
#include "allo/Support/Utils.h"
#include "allo/Translation/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/Support/raw_ostream.h"

#include "allo/Dialect/AlloDialect.h"
#include "allo/Dialect/AlloOps.h"

using namespace mlir;
using namespace allo;

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

// used for determine whether to generate C++ default types or ap_(u)int
static bool BIT_FLAG = false;

static SmallString<16> getTypeName(Type valType) {
  if (auto arrayType = valType.dyn_cast<ShapedType>())
    valType = arrayType.getElementType();

  // Handle float types.
  if (valType.isa<Float32Type>())
    return SmallString<16>("float");
  else if (valType.isa<Float64Type>())
    return SmallString<16>("double");

  // Handle integer types.
  else if (valType.isa<IndexType>())
    return SmallString<16>("int");
  else if (auto intType = valType.dyn_cast<IntegerType>()) {
    if (intType.getWidth() == 1) {
      if (!BIT_FLAG)
        return SmallString<16>("bool");
      else
        return SmallString<16>("ap_uint<1>");
    } else {
      std::string signedness = "";
      if (intType.getSignedness() == IntegerType::SignednessSemantics::Unsigned)
        signedness = "u";
      if (!BIT_FLAG) {
        switch (intType.getWidth()) {
        case 8:
        case 16:
        case 32:
        case 64:
          return SmallString<16>(signedness + "int" +
                                 std::to_string(intType.getWidth()) + "_t");
        default:
          return SmallString<16>("ap_" + signedness + "int<" +
                                 std::to_string(intType.getWidth()) + ">");
        }
      } else {
        return SmallString<16>("ap_" + signedness + "int<" +
                               std::to_string(intType.getWidth()) + ">");
      }
    }
  }

  // Handle (custom) fixed point types.
  else if (auto fixedType = valType.dyn_cast<allo::FixedType>())
    return SmallString<16>(
        "ap_fixed<" + std::to_string(fixedType.getWidth()) + ", " +
        std::to_string(fixedType.getWidth() - fixedType.getFrac()) + ">");

  else if (auto ufixedType = valType.dyn_cast<allo::UFixedType>())
    return SmallString<16>(
        "ap_ufixed<" + std::to_string(ufixedType.getWidth()) + ", " +
        std::to_string(ufixedType.getWidth() - ufixedType.getFrac()) + ">");

  else if (auto streamType = valType.dyn_cast<StreamType>())
    return SmallString<16>(
        "hls::stream< " +
        std::string(getTypeName(streamType.getBaseType()).c_str()) + " >");

  else
    assert(1 == 0 && "Got unsupported type.");

  return SmallString<16>();
}

static SmallString<16> getTypeName(Value val) {
  // Handle memref, tensor, and vector types.
  auto valType = val.getType();
  return getTypeName(valType);
}

//===----------------------------------------------------------------------===//
// ModuleEmitter Class Declaration
//===----------------------------------------------------------------------===//

namespace {
class ModuleEmitter : public AlloEmitterBase {
public:
  using operand_range = Operation::operand_range;
  explicit ModuleEmitter(AlloEmitterState &state) : AlloEmitterBase(state) {}

  /// SCF statement emitters.
  void emitScfFor(scf::ForOp op);
  void emitScfIf(scf::IfOp op);
  void emitScfYield(scf::YieldOp op);

  /// Affine statement emitters.
  void emitAffineFor(AffineForOp op);
  void emitAffineIf(AffineIfOp op);
  void emitAffineParallel(AffineParallelOp op);
  void emitAffineApply(AffineApplyOp op);
  template <typename OpType>
  void emitAffineMaxMin(OpType op, const char *syntax);
  void emitAffineLoad(AffineLoadOp op);
  void emitAffineStore(AffineStoreOp op);
  void emitAffineYield(AffineYieldOp op);

  /// Memref-related statement emitters.
  template <typename OpType> void emitAlloc(OpType op);
  void emitLoad(memref::LoadOp op);
  void emitStore(memref::StoreOp op);
  void emitGetGlobal(memref::GetGlobalOp op);
  void emitGetGlobalFixed(allo::GetGlobalFixedOp op);
  void emitGlobal(memref::GlobalOp op);
  void emitSubView(memref::SubViewOp op);

  /// Tensor-related statement emitters.
  void emitTensorExtract(tensor::ExtractOp op);
  void emitTensorInsert(tensor::InsertOp op);
  void emitDim(memref::DimOp op);
  void emitRank(memref::RankOp op);

  /// Standard expression emitters.
  void emitBinary(Operation *op, const char *syntax);
  void emitUnary(Operation *op, const char *syntax);
  void emitPower(Operation *op);
  void emitMaxMin(Operation *op, const char *syntax);

  /// Special operation emitters.
  void emitCall(func::CallOp op);
  void emitSelect(arith::SelectOp op);
  void emitConstant(arith::ConstantOp op);
  template <typename CastOpType> void emitCast(CastOpType op);
  void emitGeneralCast(UnrealizedConversionCastOp op);
  void emitGetBit(allo::GetIntBitOp op);
  void emitSetBit(allo::SetIntBitOp op);
  void emitGetSlice(allo::GetIntSliceOp op);
  void emitSetSlice(allo::SetIntSliceOp op);
  void emitBitReverse(allo::BitReverseOp op);
  void emitBitcast(arith::BitcastOp op);

  /// Stream operation emitters.
  void emitStreamConstruct(allo::StreamConstructOp op);
  void emitStreamGet(allo::StreamGetOp op);
  void emitStreamPut(allo::StreamPutOp op);

  /// Top-level MLIR module emitter.
  void emitModule(ModuleOp module);

private:
  /// C++ component emitters.
  void emitValue(Value val, unsigned rank = 0, bool isPtr = false,
                 std::string name = "");
  void emitArrayDecl(Value array, bool isFunc = false, std::string name = "");
  unsigned emitNestedLoopHead(Value val);
  void emitNestedLoopTail(unsigned rank);
  void emitInfoAndNewLine(Operation *op);

  /// MLIR component and HLS C++ pragma emitters.
  void emitBlock(Block &block);
  void emitLoopDirectives(Operation *op);
  void emitArrayDirectives(Value memref);
  void emitFunctionDirectives(func::FuncOp func, ArrayRef<Value> portList);
  void emitFunction(func::FuncOp func);
  void emitHostFunction(func::FuncOp func);
};
} // namespace

//===----------------------------------------------------------------------===//
// AffineEmitter Class
//===----------------------------------------------------------------------===//

namespace {
class AffineExprEmitter : public AlloEmitterBase,
                          public AffineExprVisitor<AffineExprEmitter> {
public:
  using operand_range = Operation::operand_range;
  explicit AffineExprEmitter(AlloEmitterState &state, unsigned numDim,
                             operand_range operands)
      : AlloEmitterBase(state), numDim(numDim), operands(operands) {}

  void visitAddExpr(AffineBinaryOpExpr expr) { emitAffineBinary(expr, "+"); }
  void visitMulExpr(AffineBinaryOpExpr expr) { emitAffineBinary(expr, "*"); }
  void visitModExpr(AffineBinaryOpExpr expr) { emitAffineBinary(expr, "%"); }
  void visitFloorDivExpr(AffineBinaryOpExpr expr) {
    emitAffineBinary(expr, "/");
  }
  void visitCeilDivExpr(AffineBinaryOpExpr expr) {
    // This is super inefficient.
    os << "(";
    visit(expr.getLHS());
    os << " + ";
    visit(expr.getRHS());
    os << " - 1) / ";
    visit(expr.getRHS());
    os << ")";
  }

  void visitConstantExpr(AffineConstantExpr expr) { os << expr.getValue(); }

  void visitDimExpr(AffineDimExpr expr) {
    os << getName(operands[expr.getPosition()]);
  }
  void visitSymbolExpr(AffineSymbolExpr expr) {
    os << getName(operands[numDim + expr.getPosition()]);
  }

  /// Affine expression emitters.
  void emitAffineBinary(AffineBinaryOpExpr expr, const char *syntax) {
    os << "(";
    if (auto constRHS = expr.getRHS().dyn_cast<AffineConstantExpr>()) {
      if ((unsigned)*syntax == (unsigned)*"*" && constRHS.getValue() == -1) {
        os << "-";
        visit(expr.getLHS());
        os << ")";
        return;
      }
      if ((unsigned)*syntax == (unsigned)*"+" && constRHS.getValue() < 0) {
        visit(expr.getLHS());
        os << " - ";
        os << -constRHS.getValue();
        os << ")";
        return;
      }
    }
    if (auto binaryRHS = expr.getRHS().dyn_cast<AffineBinaryOpExpr>()) {
      if (auto constRHS = binaryRHS.getRHS().dyn_cast<AffineConstantExpr>()) {
        if ((unsigned)*syntax == (unsigned)*"+" && constRHS.getValue() == -1 &&
            binaryRHS.getKind() == AffineExprKind::Mul) {
          visit(expr.getLHS());
          os << " - ";
          visit(binaryRHS.getLHS());
          os << ")";
          return;
        }
      }
    }
    visit(expr.getLHS());
    os << " " << syntax << " ";
    visit(expr.getRHS());
    os << ")";
  }

  void emitAffineExpr(AffineExpr expr) { visit(expr); }

private:
  unsigned numDim;
  operand_range operands;
};
} // namespace

//===----------------------------------------------------------------------===//
// StmtVisitor, ExprVisitor, and PragmaVisitor Classes
//===----------------------------------------------------------------------===//

namespace {
class StmtVisitor : public HLSCppVisitorBase<StmtVisitor, bool> {
public:
  StmtVisitor(ModuleEmitter &emitter) : emitter(emitter) {}

  using HLSCppVisitorBase::visitOp;
  /// SCF statements.
  bool visitOp(scf::ForOp op) { return emitter.emitScfFor(op), true; };
  bool visitOp(scf::IfOp op) { return emitter.emitScfIf(op), true; };
  bool visitOp(scf::ParallelOp op) { return true; };
  bool visitOp(scf::ReduceOp op) { return true; };
  bool visitOp(scf::ReduceReturnOp op) { return true; };
  bool visitOp(scf::YieldOp op) { return emitter.emitScfYield(op), true; };

  /// Affine statements.
  bool visitOp(AffineForOp op) { return emitter.emitAffineFor(op), true; }
  bool visitOp(AffineIfOp op) { return emitter.emitAffineIf(op), true; }
  bool visitOp(AffineParallelOp op) {
    return emitter.emitAffineParallel(op), true;
  }
  bool visitOp(AffineApplyOp op) { return emitter.emitAffineApply(op), true; }
  bool visitOp(AffineMaxOp op) {
    return emitter.emitAffineMaxMin<AffineMaxOp>(op, "max"), true;
  }
  bool visitOp(AffineMinOp op) {
    return emitter.emitAffineMaxMin<AffineMinOp>(op, "min"), true;
  }
  bool visitOp(AffineLoadOp op) { return emitter.emitAffineLoad(op), true; }
  bool visitOp(AffineStoreOp op) { return emitter.emitAffineStore(op), true; }
  bool visitOp(AffineYieldOp op) { return emitter.emitAffineYield(op), true; }

  /// Memref-related statements.
  bool visitOp(memref::AllocOp op) {
    return emitter.emitAlloc<memref::AllocOp>(op), true;
  }
  bool visitOp(memref::AllocaOp op) {
    return emitter.emitAlloc<memref::AllocaOp>(op), true;
  }
  bool visitOp(memref::LoadOp op) { return emitter.emitLoad(op), true; }
  bool visitOp(memref::StoreOp op) { return emitter.emitStore(op), true; }
  bool visitOp(memref::GetGlobalOp op) {
    return emitter.emitGetGlobal(op), true;
  }
  bool visitOp(allo::GetGlobalFixedOp op) {
    return emitter.emitGetGlobalFixed(op), true;
  }
  bool visitOp(memref::GlobalOp op) { return emitter.emitGlobal(op), true; }
  bool visitOp(memref::DeallocOp op) { return true; }
  bool visitOp(memref::SubViewOp op) { return emitter.emitSubView(op), true; }

  /// Tensor-related statements.
  bool visitOp(tensor::ExtractOp op) {
    return emitter.emitTensorExtract(op), true;
  }
  bool visitOp(tensor::InsertOp op) {
    return emitter.emitTensorInsert(op), true;
  }
  bool visitOp(memref::DimOp op) { return emitter.emitDim(op), true; }
  bool visitOp(memref::RankOp op) { return emitter.emitRank(op), true; }

private:
  ModuleEmitter &emitter;
};
} // namespace

namespace {
class ExprVisitor : public HLSCppVisitorBase<ExprVisitor, bool> {
public:
  ExprVisitor(ModuleEmitter &emitter) : emitter(emitter) {}

  using HLSCppVisitorBase::visitOp;
  /// Float binary expressions.
  bool visitOp(arith::CmpFOp op);
  bool visitOp(arith::AddFOp op) { return emitter.emitBinary(op, "+"), true; }
  bool visitOp(arith::SubFOp op) { return emitter.emitBinary(op, "-"), true; }
  bool visitOp(arith::MulFOp op) { return emitter.emitBinary(op, "*"), true; }
  bool visitOp(arith::DivFOp op) { return emitter.emitBinary(op, "/"), true; }
  bool visitOp(arith::RemFOp op) { return emitter.emitBinary(op, "%"), true; }

  /// Integer binary expressions.
  bool visitOp(arith::CmpIOp op);
  bool visitOp(arith::AddIOp op) { return emitter.emitBinary(op, "+"), true; }
  bool visitOp(arith::SubIOp op) { return emitter.emitBinary(op, "-"), true; }
  bool visitOp(arith::MulIOp op) { return emitter.emitBinary(op, "*"), true; }
  bool visitOp(arith::DivSIOp op) { return emitter.emitBinary(op, "/"), true; }
  bool visitOp(arith::RemSIOp op) { return emitter.emitBinary(op, "%"), true; }
  bool visitOp(arith::DivUIOp op) { return emitter.emitBinary(op, "/"), true; }
  bool visitOp(arith::RemUIOp op) { return emitter.emitBinary(op, "%"), true; }
  bool visitOp(arith::MaxSIOp op) {
    return emitter.emitMaxMin(op, "max"), true;
  }
  bool visitOp(arith::MinSIOp op) {
    return emitter.emitMaxMin(op, "min"), true;
  }
  bool visitOp(arith::MaxUIOp op) {
    return emitter.emitMaxMin(op, "max"), true;
  }
  bool visitOp(arith::MinUIOp op) {
    return emitter.emitMaxMin(op, "min"), true;
  }
  bool visitOp(arith::MaximumFOp op) {
    return emitter.emitMaxMin(op, "max"), true;
  }
  bool visitOp(arith::MinimumFOp op) {
    return emitter.emitMaxMin(op, "min"), true;
  }

  /// Logical expressions.
  bool visitOp(arith::XOrIOp op) { return emitter.emitBinary(op, "^"), true; }
  bool visitOp(arith::AndIOp op) { return emitter.emitBinary(op, "&"), true; }
  bool visitOp(arith::OrIOp op) { return emitter.emitBinary(op, "|"), true; }
  bool visitOp(arith::ShLIOp op) { return emitter.emitBinary(op, "<<"), true; }
  bool visitOp(arith::ShRSIOp op) { return emitter.emitBinary(op, ">>"), true; }
  bool visitOp(arith::ShRUIOp op) { return emitter.emitBinary(op, ">>"), true; }
  bool visitOp(allo::GetIntBitOp op) { return emitter.emitGetBit(op), true; }
  bool visitOp(allo::SetIntBitOp op) { return emitter.emitSetBit(op), true; }
  bool visitOp(allo::GetIntSliceOp op) {
    return emitter.emitGetSlice(op), true;
  }
  bool visitOp(allo::SetIntSliceOp op) {
    return emitter.emitSetSlice(op), true;
  }
  bool visitOp(allo::BitReverseOp op) {
    return emitter.emitBitReverse(op), true;
  }

  /// Unary expressions.
  bool visitOp(math::AbsFOp op) { return emitter.emitUnary(op, "abs"), true; }
  bool visitOp(math::AbsIOp op) { return emitter.emitUnary(op, "abs"), true; }
  bool visitOp(math::CeilOp op) { return emitter.emitUnary(op, "ceil"), true; }
  bool visitOp(math::CosOp op) { return emitter.emitUnary(op, "cos"), true; }
  bool visitOp(math::SinOp op) { return emitter.emitUnary(op, "sin"), true; }
  bool visitOp(math::TanhOp op) { return emitter.emitUnary(op, "tanh"), true; }
  bool visitOp(math::SqrtOp op) { return emitter.emitUnary(op, "sqrt"), true; }
  bool visitOp(math::RsqrtOp op) {
    return emitter.emitUnary(op, "1.0 / sqrt"), true;
  }
  bool visitOp(math::ExpOp op) { return emitter.emitUnary(op, "exp"), true; }
  bool visitOp(math::Exp2Op op) { return emitter.emitUnary(op, "exp2"), true; }
  bool visitOp(math::PowFOp op) { return emitter.emitPower(op), true; }
  bool visitOp(math::LogOp op) { return emitter.emitUnary(op, "log"), true; }
  bool visitOp(math::Log2Op op) { return emitter.emitUnary(op, "log2"), true; }
  bool visitOp(math::Log10Op op) {
    return emitter.emitUnary(op, "log10"), true;
  }
  bool visitOp(arith::NegFOp op) { return emitter.emitUnary(op, "-"), true; }

  /// Special operations.
  bool visitOp(func::CallOp op) { return emitter.emitCall(op), true; }
  bool visitOp(func::ReturnOp op) { return true; }
  bool visitOp(arith::SelectOp op) { return emitter.emitSelect(op), true; }
  bool visitOp(arith::ConstantOp op) { return emitter.emitConstant(op), true; }
  bool visitOp(arith::IndexCastOp op) {
    return emitter.emitCast<arith::IndexCastOp>(op), true;
  }
  bool visitOp(arith::UIToFPOp op) {
    return emitter.emitCast<arith::UIToFPOp>(op), true;
  }
  bool visitOp(arith::SIToFPOp op) {
    return emitter.emitCast<arith::SIToFPOp>(op), true;
  }
  bool visitOp(arith::FPToUIOp op) {
    return emitter.emitCast<arith::FPToUIOp>(op), true;
  }
  bool visitOp(arith::FPToSIOp op) {
    return emitter.emitCast<arith::FPToSIOp>(op), true;
  }
  bool visitOp(arith::TruncIOp op) {
    return emitter.emitCast<arith::TruncIOp>(op), true;
  }
  bool visitOp(arith::TruncFOp op) {
    return emitter.emitCast<arith::TruncFOp>(op), true;
  }
  bool visitOp(arith::ExtSIOp op) {
    return emitter.emitCast<arith::ExtSIOp>(op), true;
  }
  bool visitOp(arith::ExtUIOp op) {
    return emitter.emitCast<arith::ExtUIOp>(op), true;
  }
  bool visitOp(arith::ExtFOp op) {
    return emitter.emitCast<arith::ExtFOp>(op), true;
  }
  bool visitOp(allo::FixedToFloatOp op) {
    return emitter.emitCast<allo::FixedToFloatOp>(op), true;
  }
  bool visitOp(allo::FloatToFixedOp op) {
    return emitter.emitCast<allo::FloatToFixedOp>(op), true;
  }
  bool visitOp(allo::IntToFixedOp op) {
    return emitter.emitCast<allo::IntToFixedOp>(op), true;
  }
  bool visitOp(allo::FixedToIntOp op) {
    return emitter.emitCast<allo::FixedToIntOp>(op), true;
  }
  bool visitOp(allo::FixedToFixedOp op) {
    return emitter.emitCast<allo::FixedToFixedOp>(op), true;
  }
  bool visitOp(arith::BitcastOp op) { return emitter.emitBitcast(op), true; }
  bool visitOp(UnrealizedConversionCastOp op) {
    return emitter.emitGeneralCast(op), true;
  }

  /// Allo operations.
  bool visitOp(allo::CreateLoopHandleOp op) { return true; }
  bool visitOp(allo::CreateOpHandleOp op) { return true; }

  /// Fixed points
  bool visitOp(allo::AddFixedOp op) {
    return emitter.emitBinary(op, "+"), true;
  }
  bool visitOp(allo::SubFixedOp op) {
    return emitter.emitBinary(op, "-"), true;
  }
  bool visitOp(allo::MulFixedOp op) {
    return emitter.emitBinary(op, "*"), true;
  }
  bool visitOp(allo::DivFixedOp op) {
    return emitter.emitBinary(op, "/"), true;
  }
  bool visitOp(allo::CmpFixedOp op);
  bool visitOp(allo::ShLFixedOp op) {
    return emitter.emitBinary(op, "<<"), true;
  }
  bool visitOp(allo::ShRFixedOp op) {
    return emitter.emitBinary(op, ">>"), true;
  }
  bool visitOp(allo::MinFixedOp op) {
    return emitter.emitMaxMin(op, "min"), true;
  }
  bool visitOp(allo::MaxFixedOp op) {
    return emitter.emitMaxMin(op, "max"), true;
  }

  /// Stream operations.
  bool visitOp(allo::StreamConstructOp op) {
    return emitter.emitStreamConstruct(op), true;
  }
  bool visitOp(allo::StreamGetOp op) { return emitter.emitStreamGet(op), true; }
  bool visitOp(allo::StreamPutOp op) { return emitter.emitStreamPut(op), true; }

private:
  ModuleEmitter &emitter;
};
} // namespace

bool ExprVisitor::visitOp(arith::CmpFOp op) {
  switch (op.getPredicate()) {
  case arith::CmpFPredicate::OEQ:
  case arith::CmpFPredicate::UEQ:
    return emitter.emitBinary(op, "=="), true;
  case arith::CmpFPredicate::ONE:
  case arith::CmpFPredicate::UNE:
    return emitter.emitBinary(op, "!="), true;
  case arith::CmpFPredicate::OLT:
  case arith::CmpFPredicate::ULT:
    return emitter.emitBinary(op, "<"), true;
  case arith::CmpFPredicate::OLE:
  case arith::CmpFPredicate::ULE:
    return emitter.emitBinary(op, "<="), true;
  case arith::CmpFPredicate::OGT:
  case arith::CmpFPredicate::UGT:
    return emitter.emitBinary(op, ">"), true;
  case arith::CmpFPredicate::OGE:
  case arith::CmpFPredicate::UGE:
    return emitter.emitBinary(op, ">="), true;
  default:
    op.emitError("has unsupported compare type.");
    return false;
  }
}

bool ExprVisitor::visitOp(arith::CmpIOp op) {
  switch (op.getPredicate()) {
  case arith::CmpIPredicate::eq:
    return emitter.emitBinary(op, "=="), true;
  case arith::CmpIPredicate::ne:
    return emitter.emitBinary(op, "!="), true;
  case arith::CmpIPredicate::slt:
  case arith::CmpIPredicate::ult:
    return emitter.emitBinary(op, "<"), true;
  case arith::CmpIPredicate::sle:
  case arith::CmpIPredicate::ule:
    return emitter.emitBinary(op, "<="), true;
  case arith::CmpIPredicate::sgt:
  case arith::CmpIPredicate::ugt:
    return emitter.emitBinary(op, ">"), true;
  case arith::CmpIPredicate::sge:
  case arith::CmpIPredicate::uge:
    return emitter.emitBinary(op, ">="), true;
  }
  assert(false && "unsupported compare type");
  return false;
}

bool ExprVisitor::visitOp(allo::CmpFixedOp op) {
  switch (op.getPredicate()) {
  case allo::CmpFixedPredicate::eq:
    return emitter.emitBinary(op, "=="), true;
  case allo::CmpFixedPredicate::ne:
    return emitter.emitBinary(op, "!="), true;
  case allo::CmpFixedPredicate::slt:
  case allo::CmpFixedPredicate::ult:
    return emitter.emitBinary(op, "<"), true;
  case allo::CmpFixedPredicate::sle:
  case allo::CmpFixedPredicate::ule:
    return emitter.emitBinary(op, "<="), true;
  case allo::CmpFixedPredicate::sgt:
  case allo::CmpFixedPredicate::ugt:
    return emitter.emitBinary(op, ">"), true;
  case allo::CmpFixedPredicate::sge:
  case allo::CmpFixedPredicate::uge:
    return emitter.emitBinary(op, ">="), true;
  default:
    op.emitError("has unsupported compare type.");
    return false;
  }
}

//===----------------------------------------------------------------------===//
// ModuleEmitter Class Definition
//===----------------------------------------------------------------------===//

/// SCF statement emitters.
void ModuleEmitter::emitScfFor(scf::ForOp op) {
  indent();
  os << "for (";
  auto iterVar = op.getInductionVar();

  // Emit lower bound.
  emitValue(iterVar);
  os << " = ";
  emitValue(op.getLowerBound());
  os << "; ";

  // Emit upper bound.
  emitValue(iterVar);
  os << " < ";
  emitValue(op.getUpperBound());
  os << "; ";

  // Emit increase step.
  emitValue(iterVar);
  os << " += ";
  emitValue(op.getStep());
  os << ") {";
  emitInfoAndNewLine(op);

  addIndent();

  emitLoopDirectives(op);
  emitBlock(*op.getBody());
  reduceIndent();

  indent();
  os << "}\n";
}

void ModuleEmitter::emitScfIf(scf::IfOp op) {
  // Declare all values returned by scf::YieldOp. They will be further handled
  // by the scf::YieldOp emitter.
  for (auto result : op.getResults()) {
    if (!isDeclared(result)) {
      indent();
      if (result.getType().isa<ShapedType>())
        emitArrayDecl(result);
      else
        emitValue(result);
      os << ";\n";
    }
  }

  indent();
  os << "if (";
  emitValue(op.getCondition());
  os << ") {";
  emitInfoAndNewLine(op);

  addIndent();
  emitBlock(op.getThenRegion().front());
  reduceIndent();

  if (!op.getElseRegion().empty()) {
    indent();
    os << "} else {\n";
    addIndent();
    emitBlock(op.getElseRegion().front());
    reduceIndent();
  }

  indent();
  os << "}\n";
}

void ModuleEmitter::emitScfYield(scf::YieldOp op) {
  if (op.getNumOperands() == 0)
    return;

  // For now, only and scf::If operations will use scf::Yield to return
  // generated values.
  if (auto parentOp = dyn_cast<scf::IfOp>(op->getParentOp())) {
    unsigned resultIdx = 0;
    for (auto result : parentOp.getResults()) {
      unsigned rank = emitNestedLoopHead(result);
      indent();
      emitValue(result, rank);
      os << " = ";
      emitValue(op.getOperand(resultIdx++), rank);
      os << ";";
      emitInfoAndNewLine(op);
      emitNestedLoopTail(rank);
    }
  }
}

/// Affine statement emitters.
void ModuleEmitter::emitAffineFor(AffineForOp op) {
  indent();
  auto iterVar = op.getInductionVar();
  std::string loop_name = "";
  if (op->hasAttr("loop_name")) { // loop label
    loop_name = op->getAttr("loop_name").cast<StringAttr>().getValue().str();
    std::replace(loop_name.begin(), loop_name.end(), '.', '_');
    os << "l_";
    if (op->hasAttr("op_name")) {
      std::string op_name =
          op->getAttr("op_name").cast<StringAttr>().getValue().str();
      std::replace(op_name.begin(), op_name.end(), '.', '_');
      os << op_name << "_";
    }
    os << addName(iterVar, false, loop_name);
    os << ": ";
  }
  os << "for (";

  // Emit lower bound.
  if (op->hasAttr("loop_name")) {
    os << getTypeName(iterVar) << " ";
  }
  emitValue(iterVar, 0, false, loop_name);
  os << " = ";
  auto lowerMap = op.getLowerBoundMap();
  AffineExprEmitter lowerEmitter(state, lowerMap.getNumDims(),
                                 op.getLowerBoundOperands());
  if (lowerMap.getNumResults() == 1)
    lowerEmitter.emitAffineExpr(lowerMap.getResult(0));
  else {
    for (unsigned i = 0, e = lowerMap.getNumResults() - 1; i < e; ++i)
      os << "max(";
    lowerEmitter.emitAffineExpr(lowerMap.getResult(0));
    for (auto &expr : llvm::drop_begin(lowerMap.getResults(), 1)) {
      os << ", ";
      lowerEmitter.emitAffineExpr(expr);
      os << ")";
    }
  }
  os << "; ";

  // Emit upper bound.
  emitValue(iterVar, 0, false, loop_name);
  os << " < ";
  auto upperMap = op.getUpperBoundMap();
  AffineExprEmitter upperEmitter(state, upperMap.getNumDims(),
                                 op.getUpperBoundOperands());
  if (upperMap.getNumResults() == 1)
    upperEmitter.emitAffineExpr(upperMap.getResult(0));
  else {
    for (unsigned i = 0, e = upperMap.getNumResults() - 1; i < e; ++i)
      os << "min(";
    upperEmitter.emitAffineExpr(upperMap.getResult(0));
    for (auto &expr : llvm::drop_begin(upperMap.getResults(), 1)) {
      os << ", ";
      upperEmitter.emitAffineExpr(expr);
      os << ")";
    }
  }
  os << "; ";

  // Emit increase step.
  emitValue(iterVar, 0, false, loop_name);
  if (op.getStep() == 1)
    os << "++) {";
  else
    os << " += " << op.getStep() << ") {";
  emitInfoAndNewLine(op);

  addIndent();

  emitLoopDirectives(op);
  emitBlock(*op.getBody());
  reduceIndent();

  indent();
  os << "}\n";
}

void ModuleEmitter::emitAffineIf(AffineIfOp op) {
  // Declare all values returned by AffineYieldOp. They will be further
  // handled by the AffineYieldOp emitter.
  for (auto result : op.getResults()) {
    if (!isDeclared(result)) {
      indent();
      if (result.getType().isa<ShapedType>())
        emitArrayDecl(result);
      else
        emitValue(result);
      os << ";\n";
    }
  }

  indent();
  os << "if (";
  auto constrSet = op.getIntegerSet();
  AffineExprEmitter constrEmitter(state, constrSet.getNumDims(),
                                  op.getOperands());

  // Emit all constraints.
  unsigned constrIdx = 0;
  for (auto &expr : constrSet.getConstraints()) {
    constrEmitter.emitAffineExpr(expr);
    if (constrSet.isEq(constrIdx))
      os << " == 0";
    else
      os << " >= 0";

    if (constrIdx++ != constrSet.getNumConstraints() - 1)
      os << " && ";
  }
  os << ") {";
  emitInfoAndNewLine(op);

  addIndent();
  emitBlock(*op.getThenBlock());
  reduceIndent();

  if (op.hasElse()) {
    indent();
    os << "} else {\n";
    addIndent();
    emitBlock(*op.getElseBlock());
    reduceIndent();
  }

  indent();
  os << "}\n";
}

void ModuleEmitter::emitAffineParallel(AffineParallelOp op) {
  // Declare all values returned by AffineParallelOp. They will be further
  // handled by the AffineYieldOp emitter.
  for (auto result : op.getResults()) {
    if (!isDeclared(result)) {
      indent();
      if (result.getType().isa<ShapedType>())
        emitArrayDecl(result);
      else
        emitValue(result);
      os << ";\n";
    }
  }

  auto steps = getIntArrayAttrValue(op, op.getStepsAttrName());
  for (unsigned i = 0, e = op.getNumDims(); i < e; ++i) {
    indent();
    os << "for (";
    auto iterVar = op.getBody()->getArgument(i);

    // Emit lower bound.
    emitValue(iterVar);
    os << " = ";
    auto lowerMap = op.getLowerBoundsValueMap().getAffineMap();
    AffineExprEmitter lowerEmitter(state, lowerMap.getNumDims(),
                                   op.getLowerBoundsOperands());
    lowerEmitter.emitAffineExpr(lowerMap.getResult(i));
    os << "; ";

    // Emit upper bound.
    emitValue(iterVar);
    os << " < ";
    auto upperMap = op.getUpperBoundsValueMap().getAffineMap();
    AffineExprEmitter upperEmitter(state, upperMap.getNumDims(),
                                   op.getUpperBoundsOperands());
    upperEmitter.emitAffineExpr(upperMap.getResult(i));
    os << "; ";

    // Emit increase step.
    emitValue(iterVar);
    os << " += " << steps[i] << ") {";
    emitInfoAndNewLine(op);

    addIndent();
  }

  emitBlock(*op.getBody());

  for (unsigned i = 0, e = op.getNumDims(); i < e; ++i) {
    reduceIndent();

    indent();
    os << "}\n";
  }
}

void ModuleEmitter::emitAffineApply(AffineApplyOp op) {
  indent();
  emitValue(op.getResult());
  os << " = ";
  auto affineMap = op.getAffineMap();
  AffineExprEmitter(state, affineMap.getNumDims(), op.getOperands())
      .emitAffineExpr(affineMap.getResult(0));
  os << ";";
  emitInfoAndNewLine(op);
}

template <typename OpType>
void ModuleEmitter::emitAffineMaxMin(OpType op, const char *syntax) {
  indent();
  emitValue(op.getResult());
  os << " = ";
  auto affineMap = op.getAffineMap();
  AffineExprEmitter affineEmitter(state, affineMap.getNumDims(),
                                  op.getOperands());
  for (unsigned i = 0, e = affineMap.getNumResults() - 1; i < e; ++i)
    os << syntax << "(";
  affineEmitter.emitAffineExpr(affineMap.getResult(0));
  for (auto &expr : llvm::drop_begin(affineMap.getResults(), 1)) {
    os << ", ";
    affineEmitter.emitAffineExpr(expr);
    os << ")";
  }
  os << ";";
  emitInfoAndNewLine(op);
}

void ModuleEmitter::emitAffineLoad(AffineLoadOp op) {
  indent();
  std::string load_from_name = "";
  if (op->hasAttr("from")) {
    load_from_name = op->getAttr("from").cast<StringAttr>().getValue().str();
  }
  Value result = op.getResult();
  fixUnsignedType(result, op->hasAttr("unsigned"));
  emitValue(result);
  os << " = ";
  auto memref = op.getMemRef();
  emitValue(memref, 0, false, load_from_name);
  auto attr = memref.getType().dyn_cast<MemRefType>().getMemorySpace();
  auto affineMap = op.getAffineMap();
  AffineExprEmitter affineEmitter(state, affineMap.getNumDims(),
                                  op.getMapOperands());
  if (attr &&
      attr.cast<StringAttr>().getValue().str().substr(0, 6) == "stream") {
    auto attr_str = attr.cast<StringAttr>().getValue().str();
    int S_index = attr_str.find("S"); // spatial
    int T_index = attr_str.find("T"); // temporal
    if (S_index != -1 && T_index != -1) {
      auto st_str = attr_str.substr(S_index, T_index - S_index + 1);
      std::reverse(st_str.begin(), st_str.end());
      auto results = affineMap.getResults();
      st_str = st_str.substr(0, results.size());
      std::reverse(st_str.begin(), st_str.end());
      for (unsigned i = 0; i < results.size(); ++i) {
        if (st_str[i] == 'S') {
          os << "[";
          affineEmitter.emitAffineExpr(results[i]);
          os << "]";
        }
      }
    }
    os << ".read(); // ";
    emitValue(memref, 0, false, load_from_name); // comment
  }
  auto arrayType = memref.getType().cast<ShapedType>();
  for (auto index : affineMap.getResults()) {
    os << "[";
    affineEmitter.emitAffineExpr(index);
    os << "]";
  }
  os << ";";
  emitInfoAndNewLine(op);
}

void ModuleEmitter::emitAffineStore(AffineStoreOp op) {
  indent();
  std::string store_to_name = "";
  if (op->hasAttr("to")) {
    store_to_name = op->getAttr("to").cast<StringAttr>().getValue().str();
  }
  auto memref = op.getMemRef();
  emitValue(memref, 0, false, store_to_name);
  auto attr = memref.getType().dyn_cast<MemRefType>().getMemorySpace();
  auto affineMap = op.getAffineMap();
  AffineExprEmitter affineEmitter(state, affineMap.getNumDims(),
                                  op.getMapOperands());
  if (attr &&
      attr.cast<StringAttr>().getValue().str().substr(0, 6) == "stream") {
    auto attr_str = attr.cast<StringAttr>().getValue().str();
    int S_index = attr_str.find("S"); // spatial
    int T_index = attr_str.find("T"); // temporal
    if (S_index != -1 && T_index != -1) {
      auto st_str = attr_str.substr(S_index, T_index - S_index + 1);
      std::reverse(st_str.begin(), st_str.end());
      auto results = affineMap.getResults();
      st_str = st_str.substr(0, results.size());
      std::reverse(st_str.begin(), st_str.end());
      for (unsigned i = 0; i < results.size(); ++i) {
        if (st_str[i] == 'S') {
          os << "[";
          affineEmitter.emitAffineExpr(results[i]);
          os << "]";
        }
      }
    }
    os << ".write(";
    emitValue(op.getValueToStore());
    os << "); // ";
    emitValue(memref, 0, false, store_to_name); // comment
  }
  auto arrayType = memref.getType().cast<ShapedType>();
  for (auto index : affineMap.getResults()) {
    os << "[";
    affineEmitter.emitAffineExpr(index);
    os << "]";
  }
  os << " = ";
  emitValue(op.getValueToStore());
  os << ";";
  emitInfoAndNewLine(op);
}

// TODO: For now, all values created in the AffineIf region will be declared
// in the generated C++. However, values which will be returned by affine
// yield operation should not be declared again. How to "bind" the pair of
// values inside/outside of AffineIf region needs to be considered.
void ModuleEmitter::emitAffineYield(AffineYieldOp op) {
  if (op.getNumOperands() == 0)
    return;

  // For now, only AffineParallel and AffineIf operations will use
  // AffineYield to return generated values.
  if (auto parentOp = dyn_cast<AffineIfOp>(op->getParentOp())) {
    unsigned resultIdx = 0;
    for (auto result : parentOp.getResults()) {
      unsigned rank = emitNestedLoopHead(result);
      indent();
      emitValue(result, rank);
      os << " = ";
      emitValue(op.getOperand(resultIdx++), rank);
      os << ";";
      emitInfoAndNewLine(op);
      emitNestedLoopTail(rank);
    }
  } else if (auto parentOp = dyn_cast<AffineParallelOp>(op->getParentOp())) {
    indent();
    os << "if (";
    unsigned ivIdx = 0;
    for (auto iv : parentOp.getBody()->getArguments()) {
      emitValue(iv);
      os << " == 0";
      if (ivIdx++ != parentOp.getBody()->getNumArguments() - 1)
        os << " && ";
    }
    os << ") {\n";

    // When all induction values are 0, generated values will be directly
    // assigned to the current results, correspondingly.
    addIndent();
    unsigned resultIdx = 0;
    for (auto result : parentOp.getResults()) {
      unsigned rank = emitNestedLoopHead(result);
      indent();
      emitValue(result, rank);
      os << " = ";
      emitValue(op.getOperand(resultIdx++), rank);
      os << ";";
      emitInfoAndNewLine(op);
      emitNestedLoopTail(rank);
    }
    reduceIndent();

    indent();
    os << "} else {\n";

    // Otherwise, generated values will be accumulated/reduced to the
    // current results with corresponding arith::AtomicRMWKind operations.
    addIndent();
    auto RMWAttrs =
        getIntArrayAttrValue(parentOp, parentOp.getReductionsAttrName());
    resultIdx = 0;
    for (auto result : parentOp.getResults()) {
      unsigned rank = emitNestedLoopHead(result);
      indent();
      emitValue(result, rank);
      switch ((arith::AtomicRMWKind)RMWAttrs[resultIdx]) {
      case (arith::AtomicRMWKind::addf):
      case (arith::AtomicRMWKind::addi):
        os << " += ";
        emitValue(op.getOperand(resultIdx++), rank);
        break;
      case (arith::AtomicRMWKind::assign):
        os << " = ";
        emitValue(op.getOperand(resultIdx++), rank);
        break;
      case (arith::AtomicRMWKind::maximumf):
      case (arith::AtomicRMWKind::maxs):
      case (arith::AtomicRMWKind::maxu):
        os << " = max(";
        emitValue(result, rank);
        os << ", ";
        emitValue(op.getOperand(resultIdx++), rank);
        os << ")";
        break;
      case (arith::AtomicRMWKind::minimumf):
      case (arith::AtomicRMWKind::mins):
      case (arith::AtomicRMWKind::minu):
        os << " = min(";
        emitValue(result, rank);
        os << ", ";
        emitValue(op.getOperand(resultIdx++), rank);
        os << ")";
        break;
      case (arith::AtomicRMWKind::mulf):
      case (arith::AtomicRMWKind::muli):
        os << " *= ";
        emitValue(op.getOperand(resultIdx++), rank);
        break;
      case (arith::AtomicRMWKind::ori):
        os << " |= ";
        emitValue(op.getOperand(resultIdx++), rank);
        break;
      case (arith::AtomicRMWKind::andi):
        os << " &= ";
        emitValue(op.getOperand(resultIdx++), rank);
        break;
      }
      os << ";";
      emitInfoAndNewLine(op);
      emitNestedLoopTail(rank);
    }
    reduceIndent();

    indent();
    os << "}\n";
  }
}

/// Memref-related statement emitters.
template <typename OpType> void ModuleEmitter::emitAlloc(OpType op) {
  // A declared result indicates that the memref is output of the function, and
  // has been declared in the function signature.
  if (isDeclared(op.getResult()))
    return;

  // Vivado HLS only supports static shape on-chip memory.
  if (!op.getType().hasStaticShape())
    emitError(op, "is unranked or has dynamic shape.");

  std::string name;
  if (op->hasAttr("name")) {
    auto attr = op->getAttr("name").template cast<StringAttr>();
    name = attr.getValue().str();
  }

  indent();
  Value result = op.getResult(); // memref
  fixUnsignedType(result, op->hasAttr("unsigned"));
  emitArrayDecl(result, false, name);
  os << ";";
  emitInfoAndNewLine(op);
  emitArrayDirectives(result);
}

void ModuleEmitter::emitLoad(memref::LoadOp op) {
  indent();
  Value result = op.getResult();
  fixUnsignedType(result, op->hasAttr("unsigned"));
  emitValue(result);
  os << " = ";
  auto memref = op.getMemRef();
  emitValue(memref);
  auto attr = memref.getType().dyn_cast<MemRefType>().getMemorySpace();
  if (attr &&
      attr.cast<StringAttr>().getValue().str().substr(0, 6) == "stream") {
    auto attr_str = attr.cast<StringAttr>().getValue().str();
    int S_index = attr_str.find("S"); // spatial
    int T_index = attr_str.find("T"); // temporal
    if (S_index != -1 && T_index != -1) {
      auto st_str = attr_str.substr(S_index, T_index - S_index + 1);
      std::reverse(st_str.begin(), st_str.end());
      auto indices = op.getIndices();
      st_str = st_str.substr(0, indices.size());
      std::reverse(st_str.begin(), st_str.end());
      for (unsigned i = 0; i < indices.size(); ++i) {
        if (st_str[i] == 'S') {
          os << "[";
          emitValue(indices[i]);
          os << "]";
        }
      }
    }
    os << ".read(); // ";
    emitValue(memref); // comment
  }
  for (auto index : op.getIndices()) {
    os << "[";
    emitValue(index);
    os << "]";
  }
  os << ";";
  emitInfoAndNewLine(op);
}

void ModuleEmitter::emitStore(memref::StoreOp op) {
  indent();
  auto memref = op.getMemRef();
  emitValue(memref);
  auto attr = memref.getType().dyn_cast<MemRefType>().getMemorySpace();
  if (attr &&
      attr.cast<StringAttr>().getValue().str().substr(0, 6) == "stream") {
    auto attr_str = attr.cast<StringAttr>().getValue().str();
    int S_index = attr_str.find("S"); // spatial
    int T_index = attr_str.find("T"); // temporal
    if (S_index != -1 && T_index != -1) {
      auto st_str = attr_str.substr(S_index, T_index - S_index + 1);
      std::reverse(st_str.begin(), st_str.end());
      auto indices = op.getIndices();
      st_str = st_str.substr(0, indices.size());
      std::reverse(st_str.begin(), st_str.end());
      for (unsigned i = 0; i < indices.size(); ++i) {
        if (st_str[i] == 'S') {
          os << "[";
          emitValue(indices[i]);
          os << "]";
        }
      }
    }
    os << ".write(";
    emitValue(op.getValueToStore());
    os << "); // ";
    emitValue(memref); // comment
  }
  for (auto index : op.getIndices()) {
    os << "[";
    emitValue(index);
    os << "]";
  }
  os << " = ";
  emitValue(op.getValueToStore());
  os << ";";
  emitInfoAndNewLine(op);
}

void ModuleEmitter::emitGetGlobal(memref::GetGlobalOp op) {
  indent();
  os << "// placeholder for const ";
  Value result = op.getResult();
  fixUnsignedType(result, op->hasAttr("unsigned"));
  emitValue(result, 0, false /*isPtr*/, op.getName().str());
  emitInfoAndNewLine(op);
}

void ModuleEmitter::emitGetGlobalFixed(allo::GetGlobalFixedOp op) {
  indent();
  os << "// const ";
  Value result = op.getResult();
  fixUnsignedType(result, op->hasAttr("unsigned"));
  emitValue(result, 0, false /*isPtr*/, op.getName().str());
  os << "; /* placeholder */ ";
  emitInfoAndNewLine(op);
}

void ModuleEmitter::emitGlobal(memref::GlobalOp op) {
  auto init_val = op.getInitialValue();
  if (!init_val.has_value())
    return;
  fixUnsignedType(op, op->hasAttr("unsigned"));
  auto attr = init_val.value();
  if (auto denseAttr = attr.dyn_cast<DenseElementsAttr>()) {
    indent();
    auto arrayType = op.getType().cast<ShapedType>();
    auto type = arrayType.getElementType();
    if (op->hasAttr("constant")) {
      os << "const ";
    }
    os << getTypeName(type);
    os << " " << op.getSymName();
    for (auto &shape : arrayType.getShape())
      os << "[" << shape << "]";
    os << " = {";

    unsigned elementIdx = 0;
    for (auto element : denseAttr.getValues<Attribute>()) {
      if (type.isF32()) {
        auto value = element.cast<FloatAttr>().getValue().convertToFloat();
        if (std::isfinite(value))
          os << value;
        else if (value > 0)
          os << "INFINITY";
        else
          os << "-INFINITY";

      } else if (type.isF64()) {
        auto value = element.cast<FloatAttr>().getValue().convertToDouble();
        if (std::isfinite(value))
          os << value;
        else if (value > 0)
          os << "INFINITY";
        else
          os << "-INFINITY";

      } else if (type.isInteger(1))
        os << element.cast<BoolAttr>().getValue();
      else if (type.isIntOrIndex())
        if (op->hasAttr("unsigned")) {
          auto intType = type.dyn_cast<IntegerType>();
          os << element.cast<IntegerAttr>().getValue().getZExtValue();
          if (intType.getWidth() > 64)
            os << "ULL";
        } else {
          auto intType = type.dyn_cast<IntegerType>();
          os << element.cast<IntegerAttr>().getValue();
          if (intType.getWidth() > 64)
            os << "LL";
        }
      else
        emitError(op, "array has unsupported element type.");

      if (elementIdx++ != denseAttr.getNumElements() - 1)
        os << ", ";
    }
    os << "};";
    emitInfoAndNewLine(op);
  }
}

void ModuleEmitter::emitSubView(memref::SubViewOp op) {
  indent();
  emitArrayDecl(op.getResult(), true);
  os << " = ";
  emitValue(op.getSource());
  for (auto index : op.getOffsets()) {
    os << "[";
    emitValue(index);
    os << "]";
  }
  os << ";";
  emitInfoAndNewLine(op);
}

void ModuleEmitter::emitTensorExtract(tensor::ExtractOp op) {
  indent();
  emitValue(op.getResult());
  os << " = ";
  emitValue(op.getTensor());
  for (auto index : op.getIndices()) {
    os << "[";
    emitValue(index);
    os << "]";
  }
  os << ";";
  emitInfoAndNewLine(op);
}

void ModuleEmitter::emitTensorInsert(tensor::InsertOp op) {
  indent();
  emitValue(op.getDest());
  for (auto index : op.getIndices()) {
    os << "[";
    emitValue(index);
    os << "]";
  }
  os << " = ";
  emitValue(op.getScalar());
  os << ";";
  emitInfoAndNewLine(op);
}

void ModuleEmitter::emitDim(memref::DimOp op) {
  if (auto constOp =
          dyn_cast<arith::ConstantOp>(op.getOperand(1).getDefiningOp())) {
    auto constVal = constOp.getValue().cast<IntegerAttr>().getInt();
    auto type = op.getOperand(0).getType().cast<ShapedType>();

    if (type.hasStaticShape()) {
      if (constVal >= 0 && constVal < (int64_t)type.getShape().size()) {
        indent();
        emitValue(op.getResult());
        os << " = ";
        os << type.getShape()[constVal] << ";";
        emitInfoAndNewLine(op);
      } else
        emitError(op, "index is out of range.");
    } else
      emitError(op, "is unranked or has dynamic shape.");
  } else
    emitError(op, "index is not a constant.");
}

void ModuleEmitter::emitRank(memref::RankOp op) {
  auto type = op.getOperand().getType().cast<ShapedType>();
  if (type.hasRank()) {
    indent();
    emitValue(op.getResult());
    os << " = ";
    os << type.getRank() << ";";
    emitInfoAndNewLine(op);
  } else
    emitError(op, "is unranked.");
}

/// Standard expression emitters.
void ModuleEmitter::emitBinary(Operation *op, const char *syntax) {
  auto rank = emitNestedLoopHead(op->getResult(0));
  indent();
  Value result = op->getResult(0);
  fixUnsignedType(result, op->hasAttr("unsigned"));
  emitValue(result, rank);
  os << " = ";
  emitValue(op->getOperand(0), rank);
  os << " " << syntax << " ";
  emitValue(op->getOperand(1), rank);
  os << ";";
  emitInfoAndNewLine(op);
  emitNestedLoopTail(rank);
}

void ModuleEmitter::emitUnary(Operation *op, const char *syntax) {
  auto rank = emitNestedLoopHead(op->getResult(0));
  indent();
  Value result = op->getResult(0);
  fixUnsignedType(result, op->hasAttr("unsigned"));
  emitValue(result, rank);
  os << " = " << syntax << "(";
  emitValue(op->getOperand(0), rank);
  os << ");";
  emitInfoAndNewLine(op);
  emitNestedLoopTail(rank);
}

void ModuleEmitter::emitPower(Operation *op) {
  auto rank = emitNestedLoopHead(op->getResult(0));
  indent();
  emitValue(op->getResult(0), rank);
  os << " = pow(";
  emitValue(op->getOperand(0), rank);
  os << ", ";
  emitValue(op->getOperand(1), rank);
  os << ");";
  emitInfoAndNewLine(op);
  emitNestedLoopTail(rank);
}

/// Special operation emitters.
void ModuleEmitter::emitMaxMin(Operation *op, const char *syntax) {
  auto rank = emitNestedLoopHead(op->getResult(0));
  indent();
  Value result = op->getResult(0);
  fixUnsignedType(result, op->hasAttr("unsigned"));
  emitValue(result, rank);
  os << " = " << syntax << "(";
  emitValue(op->getOperand(0), rank);
  os << ", ";
  emitValue(op->getOperand(1), rank);
  os << ");";
  emitInfoAndNewLine(op);
  emitNestedLoopTail(rank);
}

void ModuleEmitter::emitStreamConstruct(StreamConstructOp op) {
  indent();
  Value result = op.getResult();
  fixUnsignedType(result, op->hasAttr("unsigned"));
  emitValue(result);
  if (auto shapedType = result.getType().dyn_cast<ShapedType>()) {
    for (auto shape : shapedType.getShape()) {
      os << "[" << shape << "]";
    }
  }
  os << ";\n";
  indent();
  os << "#pragma HLS stream variable=";
  emitValue(result);
  os << " depth=";
  if (result.getType().isa<StreamType>())
    os << result.getType().cast<StreamType>().getDepth();
  else {
    // array of stream
    os << result.getType()
              .cast<ShapedType>()
              .getElementType()
              .cast<StreamType>()
              .getDepth();
  }
  emitInfoAndNewLine(op);
}

void ModuleEmitter::emitStreamGet(StreamGetOp op) {
  int rank = 0;
  Value result = op.getResult();
  fixUnsignedType(result, op->hasAttr("unsigned"));
  auto stream = op->getOperand(0);
  if (stream.getType().isa<StreamType>()) {
    unsigned dimIdx = 0;
    auto streamType = stream.getType().cast<StreamType>();
    if (auto shapedType = streamType.getBaseType().dyn_cast<ShapedType>()) {
      indent();
      emitArrayDecl(result, false);
      os << ";\n";
      for (auto &shape : shapedType.getShape()) {
        indent();
        os << "for (int iv" << dimIdx << " = 0; ";
        os << "iv" << dimIdx << " < " << shape << "; ";
        os << "++iv" << dimIdx++ << ") {\n";
        addIndent();
      }
      rank = dimIdx;
    }
  }
  indent();
  emitValue(result, rank);
  os << " = ";
  emitValue(stream, 0, false);
  if (stream.getType().isa<ShapedType>()) {
    // array of stream
    auto denseArrayAttr = op->getAttrOfType<DenseI64ArrayAttr>("indices");
    for (int64_t v : denseArrayAttr.asArrayRef()) {
      os << "[" << v << "]";
    }
  }
  os << ".read();";
  if (rank > 0) {
    os << "\n";
    for (unsigned i = 0; i < rank; ++i) {
      reduceIndent();
      indent();
      os << "}\n";
    }
  }
  emitInfoAndNewLine(op);
}

void ModuleEmitter::emitStreamPut(StreamPutOp op) {
  int rank = 0;
  auto stream = op->getOperand(0);
  if (stream.getType().isa<StreamType>()) {
    unsigned dimIdx = 0;
    auto streamType = stream.getType().cast<StreamType>();
    if (auto shapedType = streamType.getBaseType().dyn_cast<ShapedType>()) {
      for (auto &shape : shapedType.getShape()) {
        indent();
        os << "for (int iv" << dimIdx << " = 0; ";
        os << "iv" << dimIdx << " < " << shape << "; ";
        os << "++iv" << dimIdx++ << ") {\n";
        addIndent();
      }
      rank = dimIdx;
    }
    indent();
    emitValue(stream, 0, false);
  } else {
    // array of stream
    indent();
    emitValue(stream, 0, false);
    auto denseArrayAttr = op->getAttrOfType<DenseI64ArrayAttr>("indices");
    for (int64_t v : denseArrayAttr.asArrayRef()) {
      os << "[" << v << "]";
    }
  }
  os << ".write(";
  emitValue(op->getOperand(1), rank);
  os << ");";
  if (rank > 0) {
    os << "\n";
    for (unsigned i = 0; i < rank; ++i) {
      reduceIndent();
      indent();
      os << "}\n";
    }
  }
  emitInfoAndNewLine(op);
}

void ModuleEmitter::emitGetBit(allo::GetIntBitOp op) {
  indent();
  Value result = op.getResult();
  fixUnsignedType(result, op->hasAttr("unsigned"));
  // generate ap_int types
  os << "ap_int<" << op.getNum().getType().getIntOrFloatBitWidth() << "> ";
  emitValue(op.getNum());
  os << "_tmp = ";
  emitValue(op.getNum());
  os << ";\n";
  // generate bit indexing
  indent();
  emitValue(result);
  os << " = ";
  emitValue(op.getNum());
  os << "_tmp[";
  emitValue(op.getIndex());
  os << "];";
  emitInfoAndNewLine(op);
}

void ModuleEmitter::emitSetBit(allo::SetIntBitOp op) {
  indent();
  // generate ap_int types
  os << "ap_int<" << op.getNum().getType().getIntOrFloatBitWidth() << "> ";
  emitValue(op.getNum());
  os << "_tmp = ";
  emitValue(op.getNum());
  os << ";\n";
  // generate bit indexing
  indent();
  emitValue(op.getNum());
  os << "_tmp[";
  emitValue(op.getIndex());
  os << "] = ";
  emitValue(op.getVal());
  os << ";";
  // write back
  indent();
  emitValue(op.getResult());
  os << " = ";
  emitValue(op.getNum());
  os << "_tmp;";
  emitInfoAndNewLine(op);
}

void ModuleEmitter::emitGetSlice(allo::GetIntSliceOp op) {
  indent();
  Value result = op.getResult();
  fixUnsignedType(result, op->hasAttr("unsigned"));
  // generate ap_int types
  os << "ap_int<" << op.getNum().getType().getIntOrFloatBitWidth() << "> ";
  emitValue(op.getNum());
  os << "_tmp = ";
  emitValue(op.getNum());
  os << ";\n";
  // generate bit slicing
  indent();
  emitValue(result);
  os << " = ";
  emitValue(op.getNum());
  os << "_tmp(";
  emitValue(op.getHi());
  os << ", ";
  emitValue(op.getLo());
  os << ");";
  emitInfoAndNewLine(op);
}

void ModuleEmitter::emitSetSlice(allo::SetIntSliceOp op) {
  indent();
  // T v;
  // v(a, b) = x;
  // c = v; // <- Need to redirect to the updated variable.
  // generate ap_int types
  os << "ap_int<" << op.getNum().getType().getIntOrFloatBitWidth() << "> ";
  emitValue(op.getNum());
  os << "_tmp = ";
  emitValue(op.getNum());
  os << ";\n";
  // generate bit slicing
  indent();
  emitValue(op.getNum());
  os << "_tmp(";
  emitValue(op.getHi());
  os << ", ";
  emitValue(op.getLo());
  os << ") = ";
  emitValue(op.getVal());
  os << ";\n";
  // write back
  indent();
  emitValue(op.getResult());
  os << " = ";
  emitValue(op.getNum());
  os << "_tmp;";
  emitInfoAndNewLine(op);
}

void ModuleEmitter::emitBitReverse(allo::BitReverseOp op) {
  indent();
  Value result = op.getResult();
  fixUnsignedType(result, op->hasAttr("unsigned"));
  emitValue(result);
  os << " = ";
  emitValue(op.getNum());
  os << ".reverse();";
  emitInfoAndNewLine(op);
}

void ModuleEmitter::emitSelect(arith::SelectOp op) {
  unsigned rank = emitNestedLoopHead(op.getResult());
  unsigned conditionRank = rank;
  if (!op.getCondition().getType().isa<ShapedType>())
    conditionRank = 0;

  indent();
  Value result = op.getResult();
  fixUnsignedType(result, op->hasAttr("unsigned"));
  emitValue(result, rank);
  os << " = ";
  emitValue(op.getCondition(), conditionRank);
  os << " ? ";
  Value true_val = op.getTrueValue();
  fixUnsignedType(true_val, op->hasAttr("unsigned"));
  os << "(" << getTypeName(true_val) << ")";
  emitValue(true_val, rank);
  os << " : ";
  Value false_val = op.getFalseValue();
  fixUnsignedType(false_val, op->hasAttr("unsigned"));
  os << "(" << getTypeName(false_val) << ")";
  emitValue(false_val, rank);
  os << ";";
  emitInfoAndNewLine(op);
  emitNestedLoopTail(rank);
}

void ModuleEmitter::emitConstant(arith::ConstantOp op) {
  // This indicates the constant type is scalar (float, integer, or bool).
  if (isDeclared(op.getResult()))
    return;

  if (auto denseAttr = op.getValue().dyn_cast<DenseElementsAttr>()) {
    indent();
    Value result = op.getResult(); // memref
    fixUnsignedType(result, op->hasAttr("unsigned"));
    emitArrayDecl(result);
    os << " = {";
    auto type = op.getResult().getType().cast<ShapedType>().getElementType();

    unsigned elementIdx = 0;
    for (auto element : denseAttr.getValues<Attribute>()) {
      if (type.isF32()) {
        auto value = element.cast<FloatAttr>().getValue().convertToFloat();
        if (std::isfinite(value))
          os << value;
        else if (value > 0)
          os << "INFINITY";
        else
          os << "-INFINITY";

      } else if (type.isF64()) {
        auto value = element.cast<FloatAttr>().getValue().convertToDouble();
        if (std::isfinite(value))
          os << value;
        else if (value > 0)
          os << "INFINITY";
        else
          os << "-INFINITY";

      } else if (type.isInteger(1))
        os << element.cast<BoolAttr>().getValue();
      else if (type.isIntOrIndex())
        os << element.cast<IntegerAttr>().getValue();
      else
        emitError(op, "array has unsupported element type.");

      if (elementIdx++ != denseAttr.getNumElements() - 1)
        os << ", ";
    }
    os << "};";
    emitInfoAndNewLine(op);
  } else
    emitError(op, "has unsupported constant type.");
}

void ModuleEmitter::emitBitcast(arith::BitcastOp op) {
  indent();
  emitValue(op.getResult());
  os << ";\n";
  indent();
  os << "union { ";
  os << getTypeName(op.getOperand());
  os << " from; ";
  os << getTypeName(op.getResult());
  os << " to;} ";
  auto name = SmallString<32>("_converter_") + getName(op.getOperand()) +
              SmallString<32>("_to_") + getName(op.getResult());
  os << name << ";\n";
  indent();
  os << name << ".from";
  os << " = ";
  emitValue(op.getOperand());
  os << ";\n";
  indent();
  emitValue(op.getResult());
  os << " = ";
  os << name << ".to;";
  emitInfoAndNewLine(op);
}

template <typename CastOpType> void ModuleEmitter::emitCast(CastOpType op) {
  indent();
  Value result = op.getResult();
  fixUnsignedType(result, op->hasAttr("unsigned"));
  emitValue(result);
  os << " = ";
  emitValue(op.getOperand());
  os << ";";
  emitInfoAndNewLine(op);
}

void ModuleEmitter::emitGeneralCast(UnrealizedConversionCastOp op) {
  indent();
  emitValue(op.getResult(0));
  os << " = ";
  emitValue(op.getOperand(0));
  os << ";";
  emitInfoAndNewLine(op);
}

void ModuleEmitter::emitCall(func::CallOp op) {
  // Handle returned value by the callee.
  for (auto result : op.getResults()) {
    if (!isDeclared(result)) {
      indent();
      if (result.getType().isa<ShapedType>())
        emitArrayDecl(result);
      else
        emitValue(result);
      os << ";\n";
    }
  }

  // Emit the function call.
  indent();
  os << op.getCallee() << "(";

  // Handle input arguments.
  unsigned argIdx = 0;
  for (auto arg : op.getOperands()) {
    emitValue(arg);

    if (argIdx++ != op.getNumOperands() - 1)
      os << ", ";
  }

  // Handle output arguments.
  for (auto result : op.getResults()) {
    // The address should be passed in for scalar result arguments.
    if (result.getType().isa<ShapedType>())
      os << ", ";
    else
      os << ", &";

    emitValue(result);
  }

  os << ");";
  emitInfoAndNewLine(op);
}

/// C++ component emitters.
void ModuleEmitter::emitValue(Value val, unsigned rank, bool isPtr,
                              std::string name) {
  assert(!(rank && isPtr) && "should be either an array or a pointer.");

  // Value has been declared before or is a constant number.
  if (isDeclared(val)) {
    os << getName(val);
    for (unsigned i = 0; i < rank; ++i)
      os << "[iv" << i << "]";
    return;
  }

  os << getTypeName(val) << " ";

  if (name == "") {
    // Add the new value to nameTable and emit its name.
    os << addName(val, isPtr);
    for (unsigned i = 0; i < rank; ++i)
      os << "[iv" << i << "]";
  } else {
    os << addName(val, isPtr, name);
  }
}

void ModuleEmitter::emitArrayDecl(Value array, bool isFunc, std::string name) {
  assert(!isDeclared(array) && "has been declared before.");

  auto arrayType = array.getType().cast<ShapedType>();
  if (arrayType.hasStaticShape()) {
    auto memref = array.getType().dyn_cast<MemRefType>();
    if (memref) {
      auto attr = memref.getMemorySpace();
      if (attr &&
          attr.cast<StringAttr>().getValue().str().substr(0, 6) == "stream") {
        // Value has been declared before or is a constant number.
        if (isDeclared(array)) {
          os << getName(array);
          return;
        }

        // print stream type
        os << "hls::stream< " << getTypeName(array) << " > ";

        auto attr_str = attr.cast<StringAttr>().getValue().str();
        int S_index = attr_str.find("S"); // spatial
        int T_index = attr_str.find("T"); // temporal
        if (isFunc &&
            !(((int)(arrayType.getShape().size()) > T_index - S_index) &&
              (T_index > S_index))) {
          os << "&"; // pass by reference, only non-array needs reference
        }

        // Add the new value to nameTable and emit its name.
        os << addName(array, /*isPtr=*/false, name);
        if ((int)(arrayType.getShape().size()) > T_index - S_index) {
          for (int i = 0; i < T_index - S_index; ++i)
            os << "[" << arrayType.getShape()[i] << "]";
        }
        // Add original array declaration as comment
        os << " /* ";
        emitValue(array, 0, false, name);
        for (auto &shape : arrayType.getShape())
          os << "[" << shape << "]";
        os << " */";
      } else {
        emitValue(array, 0, false, name);
        for (auto &shape : arrayType.getShape())
          os << "[" << shape << "]";
      }
    } else { // tensor
      emitValue(array, 0, false, name);
    }
  } else
    emitValue(array, /*rank=*/0, /*isPtr=*/true, name);
}

unsigned ModuleEmitter::emitNestedLoopHead(Value val) {
  unsigned rank = 0;

  if (auto type = val.getType().dyn_cast<ShapedType>()) {
    if (!type.hasStaticShape()) {
      emitError(val.getDefiningOp(), "is unranked or has dynamic shape.");
      return 0;
    }

    // Declare a new array.
    if (!isDeclared(val)) {
      indent();
      emitArrayDecl(val);
      os << ";\n";
    }

    // Create nested loop.
    unsigned dimIdx = 0;
    for (auto &shape : type.getShape()) {
      indent();
      os << "for (int iv" << dimIdx << " = 0; ";
      os << "iv" << dimIdx << " < " << shape << "; ";
      os << "++iv" << dimIdx++ << ") {\n";

      addIndent();
    }
    rank = type.getRank();
  }

  return rank;
}

void ModuleEmitter::emitNestedLoopTail(unsigned rank) {
  for (unsigned i = 0; i < rank; ++i) {
    reduceIndent();

    indent();
    os << "}\n";
  }
}

void ModuleEmitter::emitInfoAndNewLine(Operation *op) {
  os << "\t//";
  // Print line number.
  if (auto loc = op->getLoc().dyn_cast<FileLineColLoc>())
    os << " L" << loc.getLine();

  // // Print schedule information.
  // if (auto timing = getTiming(op))
  //   os << ", [" << timing.getBegin() << "," << timing.getEnd() << ")";

  // // Print loop information.
  // if (auto loopInfo = getLoopInfo(op))
  //   os << ", iterCycle=" << loopInfo.getIterLatency()
  //      << ", II=" << loopInfo.getMinII();

  os << "\n";
}

/// MLIR component and HLS C++ pragma emitters.
void ModuleEmitter::emitBlock(Block &block) {
  for (auto &op : block) {
    if (ExprVisitor(*this).dispatchVisitor(&op))
      continue;

    if (StmtVisitor(*this).dispatchVisitor(&op))
      continue;

    emitError(&op, "can't be correctly emitted.");
  }
}

void ModuleEmitter::emitLoopDirectives(Operation *op) {
  if (auto ii = getLoopDirective(op, "pipeline_ii")) {
    reduceIndent();
    indent();
    os << "#pragma HLS pipeline II=" << ii.cast<IntegerAttr>().getValue();
    // https://docs.xilinx.com/r/en-US/ug1399-vitis-hls/Rewinding-Pipelined-Loops-for-Performance
    if (op->hasAttr("rewind"))
      os << " rewind";
    os << "\n";
    addIndent();
  }

  if (auto factor = getLoopDirective(op, "unroll")) {
    reduceIndent();
    indent();
    auto val = factor.cast<IntegerAttr>().getValue();
    if (val == 0)
      os << "#pragma HLS unroll"
         << "\n";
    else
      os << "#pragma HLS unroll factor=" << val << "\n";
    addIndent();
  }

  if (auto dataflow = getLoopDirective(op, "dataflow")) {
    reduceIndent();
    indent();
    os << "#pragma HLS dataflow\n";
    addIndent();
  }
}

void ModuleEmitter::emitArrayDirectives(Value memref) {
  bool emitPragmaFlag = false;
  auto type = memref.getType().cast<MemRefType>();

  // streaming
  auto attr = type.getMemorySpace();
  if (attr) {
    std::string attr_str = attr.cast<StringAttr>().getValue().str();
    if (attr_str.substr(0, 6) == "stream") {
      indent();
      os << "#pragma HLS stream variable=";
      emitValue(memref);
      os << " depth=";
      int semicolon_index = attr_str.find(";");
      os << attr_str.substr(7, semicolon_index - 7);
      os << "\n";
      // if the array is a FIFO, then it cannot be further partitioned
      // so directly return
      return;
    }
  }

  if (auto layoutMap = getLayoutMap(type)) {
    // Emit array_partition pragma(s).
    SmallVector<int64_t, 8> factors;
    getPartitionFactors(type, &factors);

    for (int64_t dim = 0; dim < type.getRank(); ++dim) {
      if (!isFullyPartitioned(type, dim)) {
        if (factors[dim] != 1) {
          emitPragmaFlag = true;

          indent();
          os << "#pragma HLS array_partition";
          os << " variable=";
          emitValue(memref);

          // Emit partition type.
          if (layoutMap.getResult(dim).getKind() == AffineExprKind::FloorDiv)
            os << " block";
          else
            os << " cyclic";

          os << " dim=" << dim + 1;
          os << " factor=" << factors[dim] << "\n";
        }
      } else { // fully partitioned
        if (memref.getType().cast<ShapedType>().getShape()[dim] == 1)
          continue;

        emitPragmaFlag = true;
        indent();
        os << "#pragma HLS array_partition";
        os << " variable=";
        emitValue(memref);

        // Emit partition type.
        os << " complete";
        os << " dim=" << dim + 1 << "\n";
      }
    }
  }

  // // Emit resource pragma when the array is not DRAM kind and is not fully
  // // partitioned.
  // auto kind = MemoryKind(type.getMemorySpaceAsInt());
  // if (kind != MemoryKind::DRAM && !isFullyPartitioned(type)) {
  //   emitPragmaFlag = true;

  //   indent();
  //   os << "#pragma HLS resource";
  //   os << " variable=";
  //   emitValue(memref);

  //   os << " core=";
  //   if (kind == MemoryKind::BRAM_1P)
  //     os << "ram_1p_bram";
  //   else if (kind == MemoryKind::BRAM_S2P)
  //     os << "ram_s2p_bram";
  //   else if (kind == MemoryKind::BRAM_T2P)
  //     os << "ram_t2p_bram";
  //   else
  //     os << "ram_s2p_bram";
  //   os << "\n";
  // }

  // Emit an empty line.
  if (emitPragmaFlag)
    os << "\n";
}

void ModuleEmitter::emitFunctionDirectives(func::FuncOp func,
                                           ArrayRef<Value> portList) {
  // auto funcDirect = getFuncDirective(func);
  // if (!funcDirect)
  //   return;

  // if (funcDirect.getPipeline()) {
  //   indent();
  //   os << "#pragma HLS pipeline II=" << funcDirect.getTargetInterval() <<
  //   "\n";

  //   // An empty line.
  //   os << "\n";
  // } else if (funcDirect.getDataflow()) {
  //   indent();
  //   os << "#pragma HLS dataflow\n";

  //   // An empty line.
  //   os << "\n";
  // }

  // // Only top function should emit interface pragmas.
  // if (funcDirect.getTopFunc()) {
  //   indent();
  //   os << "#pragma HLS interface s_axilite port=return bundle=ctrl\n";

  //   for (auto &port : portList) {
  //     // Array ports and scalar ports are handled separately. Here, we only
  //     // handle MemRef types since we assume the IR has be fully bufferized.
  //     if (auto memrefType = port.getType().dyn_cast<MemRefType>()) {
  //       // Only emit interface pragma when the array is not fully
  //       partitioned. if (!isFullyPartitioned(memrefType)) {
  //         indent();
  //         os << "#pragma HLS interface";
  //         // For now, we set the offset of all m_axi interfaces as slave.
  //         if (MemoryKind(memrefType.getMemorySpaceAsInt()) ==
  //         MemoryKind::DRAM)
  //           os << " m_axi offset=slave";
  //         else
  //           os << " bram";

  //         os << " port=";
  //         emitValue(port);
  //         os << "\n";
  //       }
  //     } else {
  //       indent();
  //       os << "#pragma HLS interface s_axilite";
  //       os << " port=";

  //       // TODO: This is a temporary solution.
  //       auto name = getName(port);
  //       if (name.front() == "*"[0])
  //         name.erase(name.begin());
  //       os << name;
  //       os << " bundle=ctrl\n";
  //     }
  //   }

  //   // An empty line.
  //   os << "\n";
  if (func->hasAttr("dataflow")) {
    indent();
    os << "#pragma HLS dataflow\n";
  }

  if (func->hasAttr("inline")) {
    indent();
    os << "#pragma HLS inline\n";
  }

  // Emit other pragmas for function ports.
  for (auto &port : portList)
    if (port.getType().isa<MemRefType>())
      emitArrayDirectives(port);
  // }
}

void ModuleEmitter::emitFunction(func::FuncOp func) {
  if (func->hasAttr("bit"))
    BIT_FLAG = true;

  if (func.getBlocks().empty())
    // This is a declaration.
    return;

  if (func.getBlocks().size() > 1)
    emitError(func, "has more than one basic blocks.");

  if (func->hasAttr("top"))
    os << "/// This is top function.\n";

  // Emit function signature.
  os << "void " << func.getName() << "(\n";
  addIndent();

  // This vector is to record all ports of the function.
  SmallVector<Value, 8> portList;

  // Emit input arguments.
  unsigned argIdx = 0;
  std::vector<std::string> input_args;
  if (func->hasAttr("inputs")) {
    std::string input_names =
        func->getAttr("inputs").cast<StringAttr>().getValue().str();
    input_args = split_names(input_names);
  }
  std::string output_names;
  if (func->hasAttr("outputs")) {
    output_names = func->getAttr("outputs").cast<StringAttr>().getValue().str();
    // suppose only one output
    input_args.push_back(output_names);
  }
  std::string itypes = "";
  if (func->hasAttr("itypes"))
    itypes = func->getAttr("itypes").cast<StringAttr>().getValue().str();
  else {
    for (unsigned i = 0; i < func.getNumArguments(); ++i)
      itypes += "x";
  }
  for (auto &arg : func.getArguments()) {
    indent();
    fixUnsignedType(arg, itypes[argIdx] == 'u');
    if (arg.getType().isa<ShapedType>()) {
      if (arg.getType().cast<ShapedType>().getElementType().isa<StreamType>()) {
        auto shapedType = arg.getType().dyn_cast<ShapedType>();
        os << getTypeName(arg) << " ";
        os << addName(arg, false);
        for (auto shape : shapedType.getShape())
          os << "[" << shape << "]";
      } else if (input_args.size() == 0) {
        emitArrayDecl(arg, true);
      } else {
        emitArrayDecl(arg, true, input_args[argIdx]);
      }
    } else {
      if (arg.getType().isa<StreamType>()) {
        // need to pass by reference
        os << getTypeName(arg) << "& ";
        os << addName(arg, false);
      } else if (input_args.size() == 0) {
        emitValue(arg);
      } else {
        emitValue(arg, 0, false, input_args[argIdx]);
      }
    }

    portList.push_back(arg);
    if (argIdx++ != func.getNumArguments() - 1)
      os << ",\n";
  }

  // Emit results.
  auto args = func.getArguments();
  std::string otypes = "";
  if (func->hasAttr("otypes"))
    otypes = func->getAttr("otypes").cast<StringAttr>().getValue().str();
  else {
    for (unsigned i = 0; i < func.getNumArguments(); ++i)
      otypes += "x";
  }
  if (auto funcReturn =
          dyn_cast<func::ReturnOp>(func.front().getTerminator())) {
    unsigned idx = 0;
    for (auto result : funcReturn.getOperands()) {
      if (std::find(args.begin(), args.end(), result) == args.end()) {
        if (func.getArguments().size() > 0)
          os << ",\n";
        indent();

        // TODO: a known bug, cannot return a value twice, e.g. return %0, %0
        // : index, index. However, typically this should not happen.
        fixUnsignedType(result, otypes[idx] == 'u');
        if (result.getType().isa<ShapedType>()) {
          if (output_names != "")
            emitArrayDecl(result, true);
          else
            emitArrayDecl(result, true, output_names);
        } else {
          // In Vivado HLS, pointer indicates the value is an output.
          if (output_names != "")
            emitValue(result, /*rank=*/0, /*isPtr=*/true);
          else
            emitValue(result, /*rank=*/0, /*isPtr=*/true, output_names);
        }

        portList.push_back(result);
      }
      idx += 1;
    }
  } else
    emitError(func, "doesn't have a return operation as terminator.");

  reduceIndent();
  os << "\n) {";
  emitInfoAndNewLine(func);

  // Emit function body.
  addIndent();

  emitFunctionDirectives(func, portList);

  if (func->hasAttr("systolic")) {
    os << "#pragma scop\n";
  }
  emitBlock(func.front());
  if (func->hasAttr("systolic")) {
    os << "#pragma endscop\n";
  }

  reduceIndent();
  os << "}\n";

  // An empty line.
  os << "\n";
}

void ModuleEmitter::emitHostFunction(func::FuncOp func) {
  if (func.getBlocks().size() != 1)
    emitError(func, "has zero or more than one basic blocks.");

  os << "/// This is top function.\n";

  // Emit function signature.
  os << "int main(int argc, char **argv) {\n";
  addIndent();

  emitBlock(func.front());

  os << "  return 0;\n";
  reduceIndent();
  os << "}\n";

  // An empty line.
  os << "\n";
}

/// Top-level MLIR module emitter.
void ModuleEmitter::emitModule(ModuleOp module) {
  std::string device_header = R"XXX(
//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//
#include <algorithm>
#include <ap_axi_sdata.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <math.h>
#include <stdint.h>
using namespace std;
)XXX";

  std::string host_header = R"XXX(
//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for host
//
//===----------------------------------------------------------------------===//
// standard C/C++ headers
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <time.h>

// vivado hls headers
#include "kernel.h"
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_stream.h>

#include <ap_axi_sdata.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <math.h>
#include <stdint.h>

)XXX";

  if (module.getName().has_value() && module.getName().value() == "host") {
    os << host_header;
    for (auto op : module.getOps<func::FuncOp>()) {
      if (op.getName() == "main")
        emitHostFunction(op);
      else
        emitFunction(op);
    }
  } else {
    os << device_header;
    for (auto &op : *module.getBody()) {
      if (auto func = dyn_cast<func::FuncOp>(op))
        emitFunction(func);
      else if (auto cst = dyn_cast<memref::GlobalOp>(op))
        emitGlobal(cst);
      else
        emitError(&op, "is unsupported operation.");
    }
  }
}

//===----------------------------------------------------------------------===//
// Entry of allo-translate
//===----------------------------------------------------------------------===//

LogicalResult allo::emitVivadoHLS(ModuleOp module, llvm::raw_ostream &os) {
  AlloEmitterState state(os);
  ModuleEmitter(state).emitModule(module);
  return failure(state.encounteredError);
}

void allo::registerEmitVivadoHLSTranslation() {
  static TranslateFromMLIRRegistration toVivadoHLS(
      "emit-vivado-hls", "Emit Vivado HLS", emitVivadoHLS,
      [&](DialectRegistry &registry) {
        // clang-format off
        registry.insert<
          mlir::allo::AlloDialect,
          mlir::func::FuncDialect,
          mlir::arith::ArithDialect,
          mlir::tensor::TensorDialect,
          mlir::scf::SCFDialect,
          mlir::affine::AffineDialect,
          mlir::math::MathDialect,
          mlir::memref::MemRefDialect,
          mlir::linalg::LinalgDialect
        >();
        // clang-format on
      });
}