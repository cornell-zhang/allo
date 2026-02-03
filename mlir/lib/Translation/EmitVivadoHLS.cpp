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
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/ADT/SmallSet.h"
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
  if (auto arrayType = llvm::dyn_cast<ShapedType>(valType))
    valType = arrayType.getElementType();

  // Handle float types.
  if (llvm::isa<Float16Type>(valType))
    // Page 222:
    // https://www.amd.com/content/dam/xilinx/support/documents/sw_manuals/xilinx2020_2/ug902-vivado-high-level-synthesis.pdf
    return SmallString<16>("half");
  else if (llvm::isa<Float32Type>(valType))
    return SmallString<16>("float");
  else if (llvm::isa<Float64Type>(valType))
    return SmallString<16>("double");

  // Handle integer types.
  else if (llvm::isa<IndexType>(valType))
    return SmallString<16>("int");
  else if (auto intType = llvm::dyn_cast<IntegerType>(valType)) {
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
  else if (auto fixedType = llvm::dyn_cast<allo::FixedType>(valType))
    return SmallString<16>(
        "ap_fixed<" + std::to_string(fixedType.getWidth()) + ", " +
        std::to_string(fixedType.getWidth() - fixedType.getFrac()) + ">");

  else if (auto ufixedType = llvm::dyn_cast<allo::UFixedType>(valType))
    return SmallString<16>(
        "ap_ufixed<" + std::to_string(ufixedType.getWidth()) + ", " +
        std::to_string(ufixedType.getWidth() - ufixedType.getFrac()) + ">");

  else if (auto streamType = llvm::dyn_cast<StreamType>(valType)) {
    // Check if the base type is a shaped type (tensor/array) - stream of blocks
    if (auto baseShapedType =
            llvm::dyn_cast<ShapedType>(streamType.getBaseType())) {
      // Stream of blocks using hls::vector: Stream[elementType[dims...], depth]
      // Flatten all dimensions into a single vector size
      int64_t vectorSize = 1;
      for (auto dim : baseShapedType.getShape()) {
        vectorSize *= dim;
      }
      std::string elementTypeName =
          std::string(getTypeName(baseShapedType.getElementType()).str());
      return SmallString<16>("hls::stream< hls::vector< " + elementTypeName +
                             ", " + std::to_string(vectorSize) + " > >");
    } else {
      // Regular stream of scalars: Stream[elementType, depth]
      return SmallString<16>(
          "hls::stream< " +
          std::string(getTypeName(streamType.getBaseType()).c_str()) + " >");
    }
  }

  else
    assert(1 == 0 && "Got unsupported type.");

  return SmallString<16>();
}

/// Check if a StreamType is a stream of blocks (contains a shaped base type)
static bool isStreamOfBlocks(StreamType streamType) {
  return llvm::isa<ShapedType>(streamType.getBaseType());
}

/// Check if a Value is a function block argument (i.e., a function parameter)
/// These are pointers in the generated HLS code, not local arrays.
static bool isFunctionArgument(Value val) {
  // A value is a function argument if it has no defining operation
  // (block arguments don't have defining ops) AND it's an argument
  // of the entry block of a FuncOp.
  if (auto blockArg = dyn_cast<BlockArgument>(val)) {
    Block *block = blockArg.getOwner();
    if (block && block->isEntryBlock()) {
      if (auto funcOp = dyn_cast<func::FuncOp>(block->getParentOp())) {
        return true;
      }
    }
  }
  return false;
}

/// Check if a Value is a top-level function argument that should be linearized
static bool isTopLevelFunctionArgument(Value val, AlloEmitterState &state) {
  return state.topLevelFunctionArgs.contains(val);
}

/// Emit a linearized index expression for pointer access.
/// For a memref with shape [D0, D1, D2, ...] accessed at indices [i0, i1, i2,
/// ...], the linearized index is: i0 * (D1 * D2 * ...) + i1 * (D2 * ...) + i2 *
/// ... + ...
static void emitLinearizedAffineIndex(raw_ostream &os, AffineMap affineMap,
                                      ArrayRef<int64_t> shape, unsigned numDim,
                                      Operation::operand_range operands,
                                      AlloEmitterState &state) {
  auto results = affineMap.getResults();
  unsigned rank = results.size();

  if (rank == 0) {
    os << "[0]";
    return;
  }

  // Compute strides for row-major layout
  // stride[i] = shape[i+1] * shape[i+2] * ... * shape[rank-1]
  SmallVector<int64_t, 8> strides(rank);
  strides[rank - 1] = 1;
  for (int i = rank - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }

  // Create a temporary AffineExprEmitter to emit index expressions
  // We'll build the linearized expression manually
  os << "[";

  for (unsigned i = 0; i < rank; ++i) {
    if (i > 0)
      os << " + ";

    os << "(";
    // Emit the affine expression for this dimension
    // We need to create an AffineExprEmitter inline
    class InlineAffineEmitter : public AffineExprVisitor<InlineAffineEmitter> {
    public:
      InlineAffineEmitter(raw_ostream &os, unsigned numDim,
                          Operation::operand_range operands,
                          AlloEmitterState &state)
          : os(os), numDim(numDim), operands(operands), state(state) {}

      void visitAddExpr(AffineBinaryOpExpr expr) {
        os << "(";
        visit(expr.getLHS());
        os << " + ";
        visit(expr.getRHS());
        os << ")";
      }
      void visitMulExpr(AffineBinaryOpExpr expr) {
        os << "(";
        visit(expr.getLHS());
        os << " * ";
        visit(expr.getRHS());
        os << ")";
      }
      void visitModExpr(AffineBinaryOpExpr expr) {
        os << "(";
        visit(expr.getLHS());
        os << " % ";
        visit(expr.getRHS());
        os << ")";
      }
      void visitFloorDivExpr(AffineBinaryOpExpr expr) {
        os << "(";
        visit(expr.getLHS());
        os << " / ";
        visit(expr.getRHS());
        os << ")";
      }
      void visitCeilDivExpr(AffineBinaryOpExpr expr) {
        os << "((";
        visit(expr.getLHS());
        os << " + ";
        visit(expr.getRHS());
        os << " - 1) / ";
        visit(expr.getRHS());
        os << ")";
      }
      void visitConstantExpr(AffineConstantExpr expr) { os << expr.getValue(); }
      void visitDimExpr(AffineDimExpr expr) {
        Value operand = operands[expr.getPosition()];
        if (state.nameTable.count(operand)) {
          os << state.nameTable[operand];
        } else {
          os << "dim" << expr.getPosition();
        }
      }
      void visitSymbolExpr(AffineSymbolExpr expr) {
        Value operand = operands[numDim + expr.getPosition()];
        if (state.nameTable.count(operand)) {
          os << state.nameTable[operand];
        } else {
          os << "sym" << expr.getPosition();
        }
      }

      raw_ostream &os;
      unsigned numDim;
      Operation::operand_range operands;
      AlloEmitterState &state;
    };

    InlineAffineEmitter emitter(os, numDim, operands, state);
    emitter.visit(results[i]);
    os << ")";

    if (strides[i] > 1) {
      os << " * " << strides[i];
    }
  }

  os << "]";
}

/// Emit a linearized index expression for non-affine (memref.load/store)
/// access.
static void emitLinearizedIndex(raw_ostream &os, ValueRange indices,
                                ArrayRef<int64_t> shape,
                                AlloEmitterState &state) {
  unsigned rank = indices.size();

  if (rank == 0) {
    os << "[0]";
    return;
  }

  // Compute strides for row-major layout
  SmallVector<int64_t, 8> strides(rank);
  strides[rank - 1] = 1;
  for (int i = rank - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }

  os << "[";

  for (unsigned i = 0; i < rank; ++i) {
    if (i > 0)
      os << " + ";

    Value idx = indices[i];
    if (state.nameTable.count(idx)) {
      os << state.nameTable[idx];
    } else {
      os << "idx" << i;
    }

    if (strides[i] > 1) {
      os << " * " << strides[i];
    }
  }

  os << "]";
}

static SmallString<16> getTypeName(Value val) {
  // Handle memref, tensor, and vector types.
  auto valType = val.getType();
  return getTypeName(valType);
}

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
    if (auto constRHS = llvm::dyn_cast<AffineConstantExpr>(expr.getRHS())) {
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
    if (auto binaryRHS = llvm::dyn_cast<AffineBinaryOpExpr>(expr.getRHS())) {
      if (auto constRHS =
              llvm::dyn_cast<AffineConstantExpr>(binaryRHS.getRHS())) {
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
  StmtVisitor(allo::hls::VhlsModuleEmitter &emitter) : emitter(emitter) {}

  using HLSCppVisitorBase::visitOp;
  /// SCF statements.
  bool visitOp(scf::ForOp op) { return emitter.emitScfFor(op), true; };
  bool visitOp(scf::IfOp op) { return emitter.emitScfIf(op), true; };
  bool visitOp(scf::WhileOp op) { return emitter.emitScfWhile(op), true; };
  bool visitOp(scf::ConditionOp op) {
    return emitter.emitScfCondition(op), true;
  };
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
  bool visitOp(memref::ReshapeOp op) { return emitter.emitReshape(op), true; }

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
  allo::hls::VhlsModuleEmitter &emitter;
};
} // namespace

namespace {
class ExprVisitor : public HLSCppVisitorBase<ExprVisitor, bool> {
public:
  ExprVisitor(allo::hls::VhlsModuleEmitter &emitter) : emitter(emitter) {}

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
  bool visitOp(arith::FloorDivSIOp op) {
    return emitter.emitBinary(op, "/"), true;
  }
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
  allo::hls::VhlsModuleEmitter &emitter;
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
void allo::hls::VhlsModuleEmitter::emitScfFor(scf::ForOp op) {
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

void allo::hls::VhlsModuleEmitter::emitScfIf(scf::IfOp op) {
  // Declare all values returned by scf::YieldOp. They will be further handled
  // by the scf::YieldOp emitter.
  for (auto result : op.getResults()) {
    if (!isDeclared(result)) {
      indent();
      if (llvm::isa<ShapedType>(result.getType()))
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

namespace mlir {
namespace allo {
namespace hls {

void allo::hls::VhlsModuleEmitter::emitScfWhile(scf::WhileOp op) {
  // Declare all loop-carried values (results of while loop)
  for (auto result : op.getResults()) {
    if (!isDeclared(result)) {
      indent();
      if (llvm::isa<ShapedType>(result.getType()))
        emitArrayDecl(result);
      else
        emitValue(result);
      os << ";\n";
    }
  }

  // Initialize loop-carried variables with initial values (operands to
  // scf.while)
  unsigned operandIdx = 0;
  for (auto arg : op.getBeforeBody()->getArguments()) {
    if (operandIdx < op.getNumOperands()) {
      indent();
      emitValue(arg);
      os << " = ";
      emitValue(op.getOperand(operandIdx++));
      os << ";\n";
    }
  }

  // Emit while loop header
  indent();
  os << "while (true) {";
  emitInfoAndNewLine(op);
  addIndent();

  // Emit before block (condition check and preparation)
  // This contains computations and ends with scf.condition
  emitBlock(*op.getBeforeBody());

  // After the scf.condition updates loop vars and checks condition,
  // emit the after block (loop body)
  emitBlock(*op.getAfterBody());

  reduceIndent();
  indent();
  os << "}\n";

  // Copy final values to result variables
  // The final values are the before region's arguments after loop exit
  unsigned resultIdx = 0;
  for (auto result : op.getResults()) {
    if (resultIdx < op.getBeforeBody()->getNumArguments()) {
      indent();
      emitValue(result);
      os << " = ";
      emitValue(op.getBeforeBody()->getArgument(resultIdx++));
      os << ";\n";
    }
  }
}

void allo::hls::VhlsModuleEmitter::emitScfCondition(scf::ConditionOp op) {
  // The scf.condition op passes values to the after region.
  // First, update the after region's arguments with the values from condition
  unsigned operandIdx = 0;
  // Note: scf.while has two regions - region 0 is 'before', region 1 is 'after'
  auto afterArgs = op->getParentRegion()
                       ->getParentOp()
                       ->getRegion(1) // Get the 'after' region (index 1)
                       .front()
                       .getArguments();
  for (auto arg : afterArgs) {
    if (operandIdx < op.getNumOperands()) {
      indent();
      emitValue(arg);
      os << " = ";
      emitValue(op.getOperand(operandIdx++));
      os << ";\n";
    }
  }

  // Emit the break condition - if condition is false, break
  indent();
  os << "if (!(";
  emitValue(op.getCondition());
  os << ")) break;\n";
}

void allo::hls::VhlsModuleEmitter::emitScfYield(scf::YieldOp op) {
  if (op.getNumOperands() == 0)
    return;

  // scf::Yield can be used in scf::If or scf::While operations
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
  } else if (auto whileOp = dyn_cast<scf::WhileOp>(op->getParentOp())) {
    // In scf.while, the yield is in the after region and passes values
    // back to the before region for the next iteration
    unsigned operandIdx = 0;
    for (auto arg : whileOp.getBeforeBody()->getArguments()) {
      if (operandIdx < op.getNumOperands()) {
        // Handle array and scalar types
        unsigned rank = emitNestedLoopHead(arg);
        indent();
        emitValue(arg, rank);
        os << " = ";
        emitValue(op.getOperand(operandIdx++), rank);
        os << ";";
        emitInfoAndNewLine(op);
        emitNestedLoopTail(rank);
      }
    }
  }
}

} // namespace hls
} // namespace allo
} // namespace mlir

/// Affine statement emitters.
void allo::hls::VhlsModuleEmitter::emitAffineFor(AffineForOp op) {
  indent();
  auto iterVar = op.getInductionVar();
  std::string loop_name = "";
  if (op->hasAttr("loop_name")) { // loop label
    loop_name =
        llvm::dyn_cast<StringAttr>(op->getAttr("loop_name")).getValue().str();
    std::replace(loop_name.begin(), loop_name.end(), '.', '_');
    os << "l_";
    if (op->hasAttr("op_name")) {
      std::string op_name =
          llvm::dyn_cast<StringAttr>(op->getAttr("op_name")).getValue().str();
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

void allo::hls::VhlsModuleEmitter::emitAffineIf(AffineIfOp op) {
  // Declare all values returned by AffineYieldOp. They will be further
  // handled by the AffineYieldOp emitter.
  for (auto result : op.getResults()) {
    if (!isDeclared(result)) {
      indent();
      if (llvm::isa<ShapedType>(result.getType()))
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

void allo::hls::VhlsModuleEmitter::emitAffineParallel(AffineParallelOp op) {
  // Declare all values returned by AffineParallelOp. They will be further
  // handled by the AffineYieldOp emitter.
  for (auto result : op.getResults()) {
    if (!isDeclared(result)) {
      indent();
      if (llvm::isa<ShapedType>(result.getType()))
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

void allo::hls::VhlsModuleEmitter::emitAffineApply(AffineApplyOp op) {
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
void allo::hls::VhlsModuleEmitter::emitAffineMaxMin(OpType op,
                                                    const char *syntax) {
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

void allo::hls::VhlsModuleEmitter::emitAffineLoad(AffineLoadOp op) {
  indent();
  std::string load_from_name = "";
  if (op->hasAttr("from")) {
    load_from_name =
        llvm::dyn_cast<StringAttr>(op->getAttr("from")).getValue().str();
  }
  Value result = op.getResult();
  fixUnsignedType(result, op->hasAttr("unsigned"));
  emitValue(result);
  os << " = ";
  auto memref = op.getMemRef();
  emitValue(memref, 0, false, load_from_name);
  auto attr = llvm::dyn_cast<MemRefType>(memref.getType()).getMemorySpace();
  auto affineMap = op.getAffineMap();
  AffineExprEmitter affineEmitter(state, affineMap.getNumDims(),
                                  op.getMapOperands());
  // Check if memory space is a string attribute for streams
  auto strAttrLoad = attr ? llvm::dyn_cast<StringAttr>(attr) : nullptr;
  if (strAttrLoad && strAttrLoad.getValue().str().substr(0, 6) == "stream") {
    auto attr_str = strAttrLoad.getValue().str();
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
  auto arrayType = llvm::cast<ShapedType>(memref.getType());

  // Check if this is a top-level function argument - use linearized indexing
  // for pointers
  if (state.linearize_pointers && isTopLevelFunctionArgument(memref, state) &&
      arrayType.hasStaticShape()) {
    emitLinearizedAffineIndex(os, affineMap, arrayType.getShape(),
                              affineMap.getNumDims(), op.getMapOperands(),
                              state);
  } else {
    // Use standard multi-dimensional array access for local arrays
    for (auto index : affineMap.getResults()) {
      os << "[";
      affineEmitter.emitAffineExpr(index);
      os << "]";
    }
  }
  os << ";";
  emitInfoAndNewLine(op);
}

void allo::hls::VhlsModuleEmitter::emitAffineStore(AffineStoreOp op) {
  indent();
  std::string store_to_name = "";
  if (op->hasAttr("to")) {
    store_to_name =
        llvm::dyn_cast<StringAttr>(op->getAttr("to")).getValue().str();
  }
  auto memref = op.getMemRef();
  emitValue(memref, 0, false, store_to_name);
  auto attr = llvm::dyn_cast<MemRefType>(memref.getType()).getMemorySpace();
  auto affineMap = op.getAffineMap();
  AffineExprEmitter affineEmitter(state, affineMap.getNumDims(),
                                  op.getMapOperands());
  // Check if memory space is a string attribute for streams
  auto strAttrStore = attr ? llvm::dyn_cast<StringAttr>(attr) : nullptr;
  if (strAttrStore && strAttrStore.getValue().str().substr(0, 6) == "stream") {
    auto attr_str = strAttrStore.getValue().str();
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
  auto arrayType = llvm::cast<ShapedType>(memref.getType());

  // Check if this is a top-level function argument - use linearized indexing
  // for pointers
  if (state.linearize_pointers && isTopLevelFunctionArgument(memref, state) &&
      arrayType.hasStaticShape()) {
    emitLinearizedAffineIndex(os, affineMap, arrayType.getShape(),
                              affineMap.getNumDims(), op.getMapOperands(),
                              state);
  } else {
    // Use standard multi-dimensional array access for local arrays
    for (auto index : affineMap.getResults()) {
      os << "[";
      affineEmitter.emitAffineExpr(index);
      os << "]";
    }
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
void allo::hls::VhlsModuleEmitter::emitAffineYield(AffineYieldOp op) {
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
template <typename OpType>
void allo::hls::VhlsModuleEmitter::emitAlloc(OpType op) {
  // A declared result indicates that the memref is output of the function, and
  // has been declared in the function signature.
  if (isDeclared(op.getResult()))
    return;

  // Vivado HLS only supports static shape on-chip memory.
  if (!op.getType().hasStaticShape())
    emitError(op, "is unranked or has dynamic shape.");

  std::string name;
  if (op->hasAttr("name")) {
    auto attr = llvm::dyn_cast<StringAttr>(op->getAttr("name"));
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

void allo::hls::VhlsModuleEmitter::emitLoad(memref::LoadOp op) {
  indent();
  Value result = op.getResult();
  fixUnsignedType(result, op->hasAttr("unsigned"));
  emitValue(result);
  os << " = ";
  auto memref = op.getMemRef();
  emitValue(memref);
  auto attr = llvm::dyn_cast<MemRefType>(memref.getType()).getMemorySpace();
  // Check if memory space is a string attribute for streams
  auto strAttrMemLoad = attr ? llvm::dyn_cast<StringAttr>(attr) : nullptr;
  if (strAttrMemLoad &&
      strAttrMemLoad.getValue().str().substr(0, 6) == "stream") {
    auto attr_str = strAttrMemLoad.getValue().str();
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

  auto arrayType = llvm::cast<ShapedType>(memref.getType());

  // Check if this is a top-level function argument - use linearized indexing
  // for pointers
  if (state.linearize_pointers && isTopLevelFunctionArgument(memref, state) &&
      arrayType.hasStaticShape()) {
    emitLinearizedIndex(os, op.getIndices(), arrayType.getShape(), state);
  } else {
    // Use standard multi-dimensional array access for local arrays
    for (auto index : op.getIndices()) {
      os << "[";
      emitValue(index);
      os << "]";
    }
  }
  os << ";";
  emitInfoAndNewLine(op);
}

void allo::hls::VhlsModuleEmitter::emitStore(memref::StoreOp op) {
  indent();
  auto memref = op.getMemRef();
  emitValue(memref);
  auto attr = llvm::dyn_cast<MemRefType>(memref.getType()).getMemorySpace();
  // Check if memory space is a string attribute for streams
  auto strAttrMemStore = attr ? llvm::dyn_cast<StringAttr>(attr) : nullptr;
  if (strAttrMemStore &&
      strAttrMemStore.getValue().str().substr(0, 6) == "stream") {
    auto attr_str = strAttrMemStore.getValue().str();
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

  auto arrayType = llvm::cast<ShapedType>(memref.getType());

  // Check if this is a top-level function argument - use linearized indexing
  // for pointers
  if (state.linearize_pointers && isTopLevelFunctionArgument(memref, state) &&
      arrayType.hasStaticShape()) {
    emitLinearizedIndex(os, op.getIndices(), arrayType.getShape(), state);
  } else {
    // Use standard multi-dimensional array access for local arrays
    for (auto index : op.getIndices()) {
      os << "[";
      emitValue(index);
      os << "]";
    }
  }
  os << " = ";
  emitValue(op.getValueToStore());
  os << ";";
  emitInfoAndNewLine(op);
}

void allo::hls::VhlsModuleEmitter::emitGetGlobal(memref::GetGlobalOp op) {
  indent();
  os << "// placeholder for const ";
  Value result = op.getResult();
  fixUnsignedType(result, op->hasAttr("unsigned"));
  emitValue(result, 0, false /*isPtr*/, op.getName().str());
  emitInfoAndNewLine(op);
}

void allo::hls::VhlsModuleEmitter::emitGetGlobalFixed(
    allo::GetGlobalFixedOp op) {
  indent();
  os << "// const ";
  Value result = op.getResult();
  fixUnsignedType(result, op->hasAttr("unsigned"));
  emitValue(result, 0, false /*isPtr*/, op.getName().str());
  os << "; /* placeholder */ ";
  emitInfoAndNewLine(op);
}

void allo::hls::VhlsModuleEmitter::emitGlobal(memref::GlobalOp op) {
  auto init_val = op.getInitialValue();
  if (!init_val.has_value())
    return;
  fixUnsignedType(op, op->hasAttr("unsigned"));
  auto attr = init_val.value();
  if (auto denseAttr = llvm::dyn_cast<DenseElementsAttr>(attr)) {
    indent();
    auto arrayType = llvm::dyn_cast<ShapedType>(op.getType());
    auto type = arrayType.getElementType();
    // Check for static attribute or stateful variable naming pattern
    bool isStatic = op->hasAttr("static");
    if (!isStatic) {
      // Check if symbol name contains "__stateful_" pattern (stateful
      // variables)
      std::string symName = op.getSymName().str();
      if (symName.find("__stateful_") != std::string::npos) {
        isStatic = true;
      }
    }
    if (isStatic) {
      os << "static ";
    }
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
        auto value =
            llvm::dyn_cast<FloatAttr>(element).getValue().convertToFloat();
        if (std::isfinite(value))
          os << value;
        else if (value > 0)
          os << "INFINITY";
        else
          os << "-INFINITY";

      } else if (type.isF64()) {
        auto value =
            llvm::dyn_cast<FloatAttr>(element).getValue().convertToDouble();
        if (std::isfinite(value))
          os << value;
        else if (value > 0)
          os << "INFINITY";
        else
          os << "-INFINITY";

      } else if (type.isInteger(1))
        os << llvm::dyn_cast<BoolAttr>(element).getValue();
      else if (type.isIntOrIndex())
        if (op->hasAttr("unsigned")) {
          auto intType = llvm::dyn_cast<IntegerType>(type);
          os << llvm::dyn_cast<IntegerAttr>(element).getValue().getZExtValue();
          if (intType.getWidth() > 64)
            os << "ULL";
        } else {
          auto intType = llvm::dyn_cast<IntegerType>(type);
          os << llvm::dyn_cast<IntegerAttr>(element).getValue();
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

void allo::hls::VhlsModuleEmitter::emitSubView(memref::SubViewOp op) {
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

void allo::hls::VhlsModuleEmitter::emitTensorExtract(tensor::ExtractOp op) {
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

void allo::hls::VhlsModuleEmitter::emitTensorInsert(tensor::InsertOp op) {
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

void allo::hls::VhlsModuleEmitter::emitDim(memref::DimOp op) {
  if (auto constOp =
          dyn_cast<arith::ConstantOp>(op.getOperand(1).getDefiningOp())) {
    auto constVal = llvm::dyn_cast<IntegerAttr>(constOp.getValue()).getInt();
    auto type = llvm::dyn_cast<ShapedType>(op.getOperand(0).getType());

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

void allo::hls::VhlsModuleEmitter::emitRank(memref::RankOp op) {
  auto type = llvm::dyn_cast<ShapedType>(op.getOperand().getType());
  if (type.hasRank()) {
    indent();
    emitValue(op.getResult());
    os << " = ";
    os << type.getRank() << ";";
    emitInfoAndNewLine(op);
  } else
    emitError(op, "is unranked.");
}

/// Special operation emitters.
void allo::hls::VhlsModuleEmitter::emitStreamConstruct(StreamConstructOp op) {
  Value result = op.getResult();
  fixUnsignedType(result, op->hasAttr("unsigned"));

  // Check if this is a stream of blocks (tensor base type)
  if (auto streamType = llvm::dyn_cast<StreamType>(result.getType())) {
    if (auto baseShapedType =
            llvm::dyn_cast<ShapedType>(streamType.getBaseType())) {
      // Stream of blocks using hls::vector: Stream[elementType[dims...], depth]
      std::string varName = std::string(addName(result, false).str());

      // Compute flattened vector size
      int64_t vectorSize = 1;
      for (auto dim : baseShapedType.getShape()) {
        vectorSize *= dim;
      }

      // Emit comment describing the block structure
      indent();
      os << "// Stream of vectors: each vector packs "
         << getTypeName(baseShapedType.getElementType()) << " array";
      for (auto dim : baseShapedType.getShape()) {
        os << "[" << dim << "]";
      }
      os << " into hls::vector<" << getTypeName(baseShapedType.getElementType())
         << ", " << vectorSize << ">\n";

      // Emit the stream declaration with vector type
      indent();
      os << "hls::stream< hls::vector< "
         << getTypeName(baseShapedType.getElementType()) << ", " << vectorSize
         << " > > " << varName << ";\n";

      // Emit depth pragma
      indent();
      os << "#pragma HLS stream variable=" << varName
         << " depth=" << streamType.getDepth();
      emitInfoAndNewLine(op);
      return;
    }
  }

  // Fall back to regular stream handling for scalar streams
  indent();
  emitValue(result);
  if (auto shapedType = llvm::dyn_cast<ShapedType>(result.getType())) {
    for (auto shape : shapedType.getShape()) {
      os << "[" << shape << "]";
    }
  }
  os << ";\n";
  indent();
  os << "#pragma HLS stream variable=";
  emitValue(result);
  os << " depth=";
  if (llvm::isa<StreamType>(result.getType()))
    os << llvm::dyn_cast<StreamType>(result.getType()).getDepth();
  else {
    // array of stream
    os << llvm::dyn_cast<StreamType>(
              llvm::dyn_cast<ShapedType>(result.getType()).getElementType())
              .getDepth();
  }
  emitInfoAndNewLine(op);
}

void allo::hls::VhlsModuleEmitter::emitStreamGet(StreamGetOp op) {
  Value result = op.getResult();
  fixUnsignedType(result, op->hasAttr("unsigned"));
  auto stream = op->getOperand(0);

  StreamType streamType = nullptr;
  if (llvm::isa<StreamType>(stream.getType())) {
    streamType = llvm::dyn_cast<StreamType>(stream.getType());
  }

  if (streamType && isStreamOfBlocks(streamType)) {
    auto baseShapedType = llvm::dyn_cast<ShapedType>(streamType.getBaseType());
    std::string streamName = std::string(getName(stream).str());
    std::string resultName = std::string(addName(result, false).str());

    // Compute flattened vector size
    int64_t vectorSize = 1;
    for (auto dim : baseShapedType.getShape()) {
      vectorSize *= dim;
    }

    // 1. Declare the local result array
    indent();
    os << getTypeName(baseShapedType.getElementType()) << " " << resultName;
    for (auto dim : baseShapedType.getShape()) {
      os << "[" << dim << "]";
    }
    os << ";\n";

    // 2. Create a scope for the vector read
    indent();
    os << "{\n";
    addIndent();

    // 3. Read vector from stream
    indent();
    os << "hls::vector< " << getTypeName(baseShapedType.getElementType())
       << ", " << vectorSize << " > _vec = " << streamName << ".read();\n";

    // 4. Generate nested loops to unpack vector elements into local array
    unsigned dimIdx = 0;
    for (auto dim : baseShapedType.getShape()) {
      indent();
      os << "for (int _iv" << dimIdx << " = 0; _iv" << dimIdx << " < " << dim
         << "; ++_iv" << dimIdx++ << ") {\n";
      addIndent();
    }

    // Compute linearized index for vector access
    indent();
    os << resultName;
    for (unsigned i = 0; i < baseShapedType.getRank(); ++i) {
      os << "[_iv" << i << "]";
    }
    os << " = _vec[";
    // Build linearized index expression
    unsigned rank = baseShapedType.getRank();
    for (unsigned i = 0; i < rank; ++i) {
      if (i > 0)
        os << " + ";
      os << "_iv" << i;
      // Multiply by stride (product of remaining dimensions)
      for (unsigned j = i + 1; j < rank; ++j) {
        os << " * " << baseShapedType.getDimSize(j);
      }
    }
    os << "];\n";

    // Close loops
    for (unsigned i = 0; i < baseShapedType.getRank(); ++i) {
      reduceIndent();
      indent();
      os << "}\n";
    }

    // Close scope
    reduceIndent();
    indent();
    os << "}";
    emitInfoAndNewLine(op);
    return;
  }

  // Fallback logic for regular scalar streams
  int rank = 0;
  if (llvm::isa<StreamType>(stream.getType())) {
    unsigned dimIdx = 0;
    auto scalarStreamType = llvm::dyn_cast<StreamType>(stream.getType());
    if (auto shapedType =
            llvm::dyn_cast<ShapedType>(scalarStreamType.getBaseType())) {
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
  if (llvm::isa<ShapedType>(stream.getType())) {
    auto denseArrayAttr = op->getAttrOfType<DenseI64ArrayAttr>("indices");
    for (int64_t v : denseArrayAttr.asArrayRef())
      os << "[" << v << "]";
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

void allo::hls::VhlsModuleEmitter::emitStreamPut(StreamPutOp op) {
  auto stream = op->getOperand(0);
  auto value = op->getOperand(1);

  StreamType streamType = nullptr;
  if (llvm::isa<StreamType>(stream.getType())) {
    streamType = llvm::dyn_cast<StreamType>(stream.getType());
  }

  if (streamType && isStreamOfBlocks(streamType)) {
    auto baseShapedType = llvm::dyn_cast<ShapedType>(streamType.getBaseType());
    std::string streamName = std::string(getName(stream).str());
    std::string valueName = std::string(getName(value).str());

    // Compute flattened vector size
    int64_t vectorSize = 1;
    for (auto dim : baseShapedType.getShape()) {
      vectorSize *= dim;
    }

    // 1. Create scope for vector write
    indent();
    os << "{\n";
    addIndent();

    // 2. Declare vector to pack data into
    indent();
    os << "hls::vector< " << getTypeName(baseShapedType.getElementType())
       << ", " << vectorSize << " > _vec;\n";

    // 3. Generate nested loops to pack array elements into vector
    unsigned dimIdx = 0;
    for (auto dim : baseShapedType.getShape()) {
      indent();
      os << "for (int _iv" << dimIdx << " = 0; _iv" << dimIdx << " < " << dim
         << "; ++_iv" << dimIdx++ << ") {\n";
      addIndent();
    }

    // Compute linearized index for vector access
    indent();
    os << "_vec[";
    unsigned rank = baseShapedType.getRank();
    for (unsigned i = 0; i < rank; ++i) {
      if (i > 0)
        os << " + ";
      os << "_iv" << i;
      // Multiply by stride (product of remaining dimensions)
      for (unsigned j = i + 1; j < rank; ++j) {
        os << " * " << baseShapedType.getDimSize(j);
      }
    }
    os << "] = " << valueName;
    for (unsigned i = 0; i < baseShapedType.getRank(); ++i) {
      os << "[_iv" << i << "]";
    }
    os << ";\n";

    // Close loops
    for (unsigned i = 0; i < baseShapedType.getRank(); ++i) {
      reduceIndent();
      indent();
      os << "}\n";
    }

    // 4. Write vector to stream
    indent();
    os << streamName << ".write(_vec);\n";

    // Close scope
    reduceIndent();
    indent();
    os << "}";
    emitInfoAndNewLine(op);
    return;
  }

  // Fallback logic for regular scalar streams
  int rank = 0;
  if (llvm::isa<StreamType>(stream.getType())) {
    unsigned dimIdx = 0;
    auto scalarStreamType = llvm::dyn_cast<StreamType>(stream.getType());
    if (auto shapedType =
            llvm::dyn_cast<ShapedType>(scalarStreamType.getBaseType())) {
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
    indent();
    emitValue(stream, 0, false);
    auto denseArrayAttr = op->getAttrOfType<DenseI64ArrayAttr>("indices");
    for (int64_t v : denseArrayAttr.asArrayRef())
      os << "[" << v << "]";
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

void allo::hls::VhlsModuleEmitter::emitGetBit(allo::GetIntBitOp op) {
  indent();
  Value result = op.getResult();
  fixUnsignedType(result, op->hasAttr("unsigned"));
  emitValue(result);
  os << ";\n";
  indent();
  // generate ap_int types
  os << "ap_int<" << op.getNum().getType().getIntOrFloatBitWidth() << "> ";
  os << getName(result);
  os << "_tmp = ";
  emitValue(op.getNum());
  os << ";\n";
  // generate bit indexing
  indent();
  emitValue(result);
  os << " = ";
  os << getName(result);
  os << "_tmp[";
  emitValue(op.getIndex());
  os << "];";
  emitInfoAndNewLine(op);
}

void allo::hls::VhlsModuleEmitter::emitSetBit(allo::SetIntBitOp op) {
  indent();
  emitValue(op.getResult());
  os << ";\n";
  // generate ap_int types
  indent();
  os << "ap_int<" << op.getNum().getType().getIntOrFloatBitWidth() << "> ";
  os << getName(op.getResult());
  os << "_tmp = ";
  emitValue(op.getNum());
  os << ";\n";
  // generate bit indexing
  indent();
  os << getName(op.getResult());
  os << "_tmp[";
  emitValue(op.getIndex());
  os << "] = ";
  emitValue(op.getVal());
  os << ";";
  // write back
  indent();
  emitValue(op.getResult());
  os << " = ";
  os << getName(op.getResult());
  os << "_tmp;";
  emitInfoAndNewLine(op);
}

void allo::hls::VhlsModuleEmitter::emitGetSlice(allo::GetIntSliceOp op) {
  indent();
  Value result = op.getResult();
  emitValue(result);
  os << ";\n";
  fixUnsignedType(result, op->hasAttr("unsigned"));
  // generate ap_int types
  indent();
  os << "ap_int<" << op.getNum().getType().getIntOrFloatBitWidth() << "> ";
  os << getName(result);
  os << "_tmp = ";
  emitValue(op.getNum());
  os << ";\n";
  // generate bit slicing
  indent();
  emitValue(result);
  os << " = ";
  os << getName(result);
  os << "_tmp(";
  emitValue(op.getHi());
  os << ", ";
  emitValue(op.getLo());
  os << ");";
  emitInfoAndNewLine(op);
}

void allo::hls::VhlsModuleEmitter::emitSetSlice(allo::SetIntSliceOp op) {
  indent();
  // T v;
  // v(a, b) = x;
  // c = v; // <- Need to redirect to the updated variable.
  emitValue(op.getResult());
  os << ";\n";
  // generate ap_int types
  indent();
  os << "ap_int<" << op.getNum().getType().getIntOrFloatBitWidth() << "> ";
  os << getName(op.getResult());
  os << "_tmp = ";
  emitValue(op.getNum());
  os << ";\n";
  // generate bit slicing
  indent();
  os << getName(op.getResult());
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
  os << getName(op.getResult());
  os << "_tmp;";
  emitInfoAndNewLine(op);
}

void allo::hls::VhlsModuleEmitter::emitBitReverse(allo::BitReverseOp op) {
  indent();
  Value result = op.getResult();
  fixUnsignedType(result, op->hasAttr("unsigned"));
  emitValue(result);
  os << " = ";
  emitValue(op.getNum());
  os << ".reverse();";
  emitInfoAndNewLine(op);
}

void allo::hls::VhlsModuleEmitter::emitReshape(memref::ReshapeOp op) {
  auto array = op->getResult(0);
  assert(!isDeclared(array) && "has been declared before.");

  auto arrayType = llvm::dyn_cast<ShapedType>(array.getType());
  indent() << getTypeName(array) << " (*";

  // Add the new value to nameTable and emit its name.
  os << addName(array, false);
  os << ")";

  for (auto &shape : llvm::drop_begin(arrayType.getShape(), 1))
    os << "[" << shape << "]";

  os << " = (" << getTypeName(array) << "(*)";
  for (auto &shape : llvm::drop_begin(arrayType.getShape(), 1))
    os << "[" << shape << "]";
  os << ") ";

  emitValue(op->getOperand(0));
  os << ";";
  emitInfoAndNewLine(op);
}

void allo::hls::VhlsModuleEmitter::emitSelect(arith::SelectOp op) {
  unsigned rank = emitNestedLoopHead(op.getResult());
  unsigned conditionRank = rank;
  if (!llvm::isa<ShapedType>(op.getCondition().getType()))
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

void allo::hls::VhlsModuleEmitter::emitConstant(arith::ConstantOp op) {
  // This indicates the constant type is scalar (float, integer, or bool).
  if (isDeclared(op.getResult()))
    return;

  if (auto denseAttr = llvm::dyn_cast<DenseElementsAttr>(op.getValue())) {
    indent();
    Value result = op.getResult(); // memref
    fixUnsignedType(result, op->hasAttr("unsigned"));
    emitArrayDecl(result);
    os << " = {";
    auto type =
        llvm::dyn_cast<ShapedType>(op.getResult().getType()).getElementType();

    unsigned elementIdx = 0;
    for (auto element : denseAttr.getValues<Attribute>()) {
      if (type.isF32()) {
        auto value =
            llvm::dyn_cast<FloatAttr>(element).getValue().convertToFloat();
        if (std::isfinite(value))
          os << value;
        else if (value > 0)
          os << "INFINITY";
        else
          os << "-INFINITY";

      } else if (type.isF64()) {
        auto value =
            llvm::dyn_cast<FloatAttr>(element).getValue().convertToDouble();
        if (std::isfinite(value))
          os << value;
        else if (value > 0)
          os << "INFINITY";
        else
          os << "-INFINITY";

      } else if (type.isInteger(1))
        os << llvm::dyn_cast<BoolAttr>(element).getValue();
      else if (type.isIntOrIndex())
        os << llvm::dyn_cast<IntegerAttr>(element).getValue();
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

void allo::hls::VhlsModuleEmitter::emitBitcast(arith::BitcastOp op) {
  indent();
  Value result = op.getResult();
  fixUnsignedType(result, op->hasAttr("unsigned"));
  Value operand = op.getOperand();
  fixUnsignedType(operand, op->hasAttr("unsigned"));
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

template <typename CastOpType>
void allo::hls::VhlsModuleEmitter::emitCast(CastOpType op) {
  indent();
  Value result = op.getResult();
  fixUnsignedType(result, op->hasAttr("unsigned"));
  emitValue(result);
  os << " = ";
  emitValue(op.getOperand());
  os << ";";
  emitInfoAndNewLine(op);
}

void allo::hls::VhlsModuleEmitter::emitGeneralCast(
    UnrealizedConversionCastOp op) {
  indent();
  emitValue(op.getResult(0));
  os << " = ";
  emitValue(op.getOperand(0));
  os << ";";
  emitInfoAndNewLine(op);
}

void allo::hls::VhlsModuleEmitter::emitCall(func::CallOp op) {
  // Handle returned value by the callee.
  // For HLS C++, any function with return values needs those values
  // declared as variables and passed as pointer arguments.
  for (auto result : op.getResults()) {
    if (!isDeclared(result)) {
      indent();
      if (llvm::isa<ShapedType>(result.getType()))
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
  // For HLS C++, return values are passed as pointer arguments.
  for (auto result : op.getResults()) {
    // The address should be passed in for scalar result arguments.
    if (llvm::isa<ShapedType>(result.getType()))
      os << ", ";
    else
      os << ", &";

    emitValue(result);
  }

  os << ");";
  emitInfoAndNewLine(op);
}

/// C++ component emitters.
void allo::hls::VhlsModuleEmitter::emitValue(Value val, unsigned rank,
                                             bool isPtr, std::string name) {
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

void allo::hls::VhlsModuleEmitter::emitArrayDecl(Value array, bool isFunc,
                                                 std::string name) {
  assert(!isDeclared(array) && "has been declared before.");

  auto arrayType = llvm::dyn_cast<ShapedType>(array.getType());
  if (arrayType.hasStaticShape()) {
    auto memref = llvm::dyn_cast<MemRefType>(array.getType());
    if (memref) {
      auto attr = memref.getMemorySpace();
      // Check if memory space is a string attribute for streams
      auto strAttr = attr ? llvm::dyn_cast<StringAttr>(attr) : nullptr;
      if (strAttr && strAttr.getValue().str().substr(0, 6) == "stream") {
        // Value has been declared before or is a constant number.
        if (isDeclared(array)) {
          os << getName(array);
          return;
        }

        // print stream type
        os << "hls::stream< " << getTypeName(array) << " > ";

        auto attr_str = strAttr.getValue().str();
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

unsigned allo::hls::VhlsModuleEmitter::emitNestedLoopHead(Value val) {
  unsigned rank = 0;

  if (auto type = llvm::dyn_cast<ShapedType>(val.getType())) {
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

/// MLIR component and HLS C++ pragma emitters.
void allo::hls::VhlsModuleEmitter::emitBlock(Block &block) {
  for (auto &op : block) {
    if (ExprVisitor(*this).dispatchVisitor(&op))
      continue;

    if (StmtVisitor(*this).dispatchVisitor(&op))
      continue;

    emitError(&op, "can't be correctly emitted.");
  }
}

void allo::hls::VhlsModuleEmitter::emitLoopDirectives(Operation *op) {
  if (auto ii = getLoopDirective(op, "pipeline_ii")) {
    reduceIndent();
    indent();
    os << "#pragma HLS pipeline II="
       << llvm::dyn_cast<IntegerAttr>(ii).getValue();
    // https://docs.xilinx.com/r/en-US/ug1399-vitis-hls/Rewinding-Pipelined-Loops-for-Performance
    if (op->hasAttr("rewind"))
      os << " rewind";
    os << "\n";
    addIndent();
  }

  if (auto factor = getLoopDirective(op, "unroll")) {
    reduceIndent();
    indent();
    auto val = llvm::dyn_cast<IntegerAttr>(factor).getValue();
    if (val == 0)
      os << "#pragma HLS unroll" << "\n";
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

void allo::hls::VhlsModuleEmitter::emitArrayDirectives(Value memref) {
  bool emitPragmaFlag = false;
  auto type = llvm::dyn_cast<MemRefType>(memref.getType());

  // streaming or memory implementation
  auto attr = type.getMemorySpace();
  if (attr) {
    // Check if it's a string attribute (streaming)
    if (auto strAttr = llvm::dyn_cast<StringAttr>(attr)) {
      std::string attr_str = strAttr.getValue().str();
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
    // Check if it's an integer attribute (memory implementation)
    else if (auto intAttr = llvm::dyn_cast<IntegerAttr>(attr)) {
      int64_t memSpace = intAttr.getInt();
      if (memSpace > 0) {
        // Decode: memory_space = impl_code * 16 + storage_type_code
        int implCode = memSpace / 16;
        int storageCode = memSpace % 16;

        // Map impl_code to implementation type
        std::string implType;
        switch (implCode) {
        case 1:
          implType = "bram";
          break;
        case 2:
          implType = "uram";
          break;
        case 3:
          implType = "lutram";
          break;
        case 4:
          implType = "srl";
          break;
        default:
          implType = "";
          break; // AUTO or unknown
        }

        // Map storage_code to storage type
        std::string storageType;
        switch (storageCode) {
        case 1:
          storageType = "ram_1p";
          break;
        case 2:
          storageType = "ram_2p";
          break;
        case 3:
          storageType = "ram_t2p";
          break;
        case 4:
          storageType = "ram_1wnr";
          break;
        case 5:
          storageType = "ram_s2p";
          break;
        case 6:
          storageType = "rom_1p";
          break;
        case 7:
          storageType = "rom_2p";
          break;
        case 8:
          storageType = "rom_np";
          break;
        default:
          storageType = "";
          break;
        }

        // Emit bind_storage pragma if we have a valid implementation type
        if (!implType.empty()) {
          emitPragmaFlag = true;
          indent();
          os << "#pragma HLS bind_storage variable=";
          emitValue(memref);
          if (!storageType.empty()) {
            os << " type=" << storageType;
          }
          os << " impl=" << implType;
          os << "\n";
        }
      }
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
        if (llvm::dyn_cast<ShapedType>(memref.getType()).getShape()[dim] == 1)
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

void allo::hls::VhlsModuleEmitter::emitFunctionDirectives(
    func::FuncOp func, ArrayRef<Value> portList) {
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
    if (llvm::isa<MemRefType>(port.getType()))
      emitArrayDirectives(port);
  // }
}

void allo::hls::VhlsModuleEmitter::emitFunction(func::FuncOp func) {
  if (func->hasAttr("bit"))
    BIT_FLAG = true;

  if (func.getBlocks().empty())
    // This is a declaration.
    return;

  if (func.getBlocks().size() > 1)
    emitError(func, "has more than one basic blocks.");

  bool isTopFunction = func->hasAttr("top");
  if (isTopFunction)
    os << "/// This is top function.\n";

  // Note: Top-level function arguments are tracked in emitModule() before any
  // functions are emitted, so state.topLevelFunctionArgs is already populated

  // Validate nested function calls if linearize_pointers is enabled
  // If there are any top-level multi-dimensional arrays tracked, and this is a
  // nested function, we need to check if any are used here
  if (state.linearize_pointers && !isTopFunction &&
      !state.topLevelFunctionArgs.empty()) {
    // Check if this nested function has any multi-dimensional array arguments
    // If it does AND we have top-level multi-dim arrays, it's potentially
    // problematic
    for (auto arg : func.getArguments()) {
      if (auto shapedType = llvm::dyn_cast<ShapedType>(arg.getType())) {
        if (shapedType.hasStaticShape() && shapedType.getRank() > 1) {
          // This nested function has a multi-dimensional array parameter
          // This could be a top-level array being passed down (error) or a
          // local array (ok) Since we can't easily trace data flow at this
          // point, we emit a conservative error The safest approach: if
          // top-level has multi-dim arrays AND nested function has multi-dim
          // parameters, that's an error (conservative)
          emitError(func,
                    "nested function cannot have multi-dimensional array "
                    "arguments when "
                    "wrap_io=False (flatten=True) and the top-level function "
                    "has multi-dimensional arrays. "
                    "Top-level multi-dimensional arrays are linearized to 1D "
                    "pointers (e.g., float *v) "
                    "which cannot be passed to nested functions expecting "
                    "multi-dimensional arrays (e.g., float v[M][N]). "
                    "Solution: Use only 1D arrays as arguments to nested "
                    "functions, or enable wrap_io=True.");
          return;
        }
      }
    }
  }

  // Collect stateful globals used in this function
  std::vector<memref::GlobalOp> statefulGlobals;
  func.walk([&](memref::GetGlobalOp getGlobalOp) {
    auto globalOp =
        getGlobalOp->getParentOfType<ModuleOp>().lookupSymbol<memref::GlobalOp>(
            getGlobalOp.getName());
    if (globalOp) {
      bool isStatic = globalOp->hasAttr("static");
      if (!isStatic) {
        // Check if symbol name contains "__stateful_" pattern (stateful
        // variables)
        std::string symName = globalOp.getSymName().str();
        if (symName.find("__stateful_") != std::string::npos) {
          isStatic = true;
        }
      }
      if (isStatic) {
        // Check if we've already added this global
        bool found = false;
        for (auto &g : statefulGlobals) {
          if (g.getSymName() == globalOp.getSymName()) {
            found = true;
            break;
          }
        }
        if (!found) {
          statefulGlobals.push_back(globalOp);
        }
      }
    }
  });

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
        llvm::dyn_cast<StringAttr>(func->getAttr("inputs")).getValue().str();
    input_args = split_names(input_names);
  }
  std::string output_names;
  if (func->hasAttr("outputs")) {
    output_names =
        llvm::dyn_cast<StringAttr>(func->getAttr("outputs")).getValue().str();
    // suppose only one output
    input_args.push_back(output_names);
  }
  std::string itypes = "";
  if (func->hasAttr("itypes"))
    itypes =
        llvm::dyn_cast<StringAttr>(func->getAttr("itypes")).getValue().str();
  else {
    for (unsigned i = 0; i < func.getNumArguments(); ++i)
      itypes += "x";
  }
  for (auto &arg : func.getArguments()) {
    indent();
    fixUnsignedType(arg, itypes[argIdx] == 'u');
    if (llvm::isa<ShapedType>(arg.getType())) {
      if (llvm::isa<StreamType>(
              llvm::dyn_cast<ShapedType>(arg.getType()).getElementType())) {
        auto shapedType = llvm::dyn_cast<ShapedType>(arg.getType());
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
      if (llvm::isa<StreamType>(arg.getType())) {
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
    otypes =
        llvm::dyn_cast<StringAttr>(func->getAttr("otypes")).getValue().str();
  else {
    for (unsigned i = 0; i < func.getNumArguments(); ++i)
      otypes += "x";
  }
  if (auto funcReturn =
          llvm::dyn_cast<func::ReturnOp>(func.front().getTerminator())) {
    unsigned idx = 0;
    for (auto result : funcReturn.getOperands()) {
      if (std::find(args.begin(), args.end(), result) == args.end()) {
        if (func.getArguments().size() > 0)
          os << ",\n";
        indent();

        // TODO: a known bug, cannot return a value twice, e.g. return %0, %0
        // : index, index. However, typically this should not happen.
        fixUnsignedType(result, otypes[idx] == 'u');
        if (llvm::isa<ShapedType>(result.getType())) {
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

  // Emit stateful globals inside the function (as static variables)
  for (auto &globalOp : statefulGlobals) {
    auto init_val = globalOp.getInitialValue();
    if (!init_val.has_value())
      continue;
    fixUnsignedType(globalOp, globalOp->hasAttr("unsigned"));
    auto attr = init_val.value();
    if (auto denseAttr = llvm::dyn_cast<DenseElementsAttr>(attr)) {
      indent();
      auto arrayType = llvm::cast<ShapedType>(globalOp.getType());
      auto type = arrayType.getElementType();
      // Stateful variables are always static when inside a function
      os << "static ";
      if (globalOp->hasAttr("constant")) {
        os << "const ";
      }
      os << getTypeName(type);
      os << " " << globalOp.getSymName();
      for (auto &shape : arrayType.getShape())
        os << "[" << shape << "]";
      os << " = {";

      unsigned elementIdx = 0;
      for (auto element : denseAttr.getValues<Attribute>()) {
        if (type.isF32()) {
          auto value =
              llvm::cast<FloatAttr>(element).getValue().convertToFloat();
          if (std::isfinite(value))
            os << value;
          else if (value > 0)
            os << "INFINITY";
          else
            os << "-INFINITY";
        } else if (type.isF64()) {
          auto value =
              llvm::cast<FloatAttr>(element).getValue().convertToDouble();
          if (std::isfinite(value))
            os << value;
          else if (value > 0)
            os << "INFINITY";
          else
            os << "-INFINITY";
        } else if (type.isInteger(1))
          os << llvm::cast<BoolAttr>(element).getValue();
        else if (type.isIntOrIndex())
          if (globalOp->hasAttr("unsigned")) {
            auto intType = llvm::dyn_cast<IntegerType>(type);
            os << llvm::cast<IntegerAttr>(element).getValue().getZExtValue();
            if (intType.getWidth() > 64)
              os << "ULL";
          } else {
            auto intType = llvm::dyn_cast<IntegerType>(type);
            os << llvm::cast<IntegerAttr>(element).getValue();
            if (intType.getWidth() > 64)
              os << "LL";
          }
        else
          emitError(globalOp.getOperation(),
                    "array has unsupported element type.");

        if (elementIdx++ != denseAttr.getNumElements() - 1)
          os << ", ";
      }
      os << "};";
      emitInfoAndNewLine(globalOp.getOperation());
    }
  }

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

void allo::hls::VhlsModuleEmitter::emitHostFunction(func::FuncOp func) {
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
void allo::hls::VhlsModuleEmitter::emitModule(ModuleOp module) {
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
#include <hls_vector.h>
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
    // First pass: collect all globals and determine which are stateful
    llvm::SmallSet<StringRef, 4> statefulGlobalNames;
    for (auto &op : *module.getBody()) {
      if (auto cst = dyn_cast<memref::GlobalOp>(op)) {
        bool isStatic = cst->hasAttr("static");
        if (!isStatic) {
          // Check if symbol name contains "__stateful_" pattern (stateful
          // variables)
          std::string symName = cst.getSymName().str();
          if (symName.find("__stateful_") != std::string::npos) {
            isStatic = true;
          }
        }
        if (isStatic) {
          statefulGlobalNames.insert(cst.getSymName());
        }
      }
    }

    // Second pass: identify top-level function and track its multi-dimensional
    // arrays This must be done before emitting any functions to properly
    // validate nested functions
    if (state.linearize_pointers) {
      // Find the top function - try attribute first, then use calling pattern
      func::FuncOp topFunc = nullptr;
      llvm::SmallSet<StringRef, 4> calledFunctions;

      // First, collect all called functions
      for (auto &op : *module.getBody()) {
        if (auto func = dyn_cast<func::FuncOp>(op)) {
          func.walk([&](func::CallOp callOp) {
            calledFunctions.insert(callOp.getCallee());
          });
        }
      }

      // Now find the top function
      for (auto &op : *module.getBody()) {
        if (auto func = dyn_cast<func::FuncOp>(op)) {
          // Check if it has the "top" attribute (set by HLS backend)
          if (func->hasAttr("top")) {
            topFunc = func;
            break;
          }

          // If no function has been marked as top yet, the top function is the
          // one that is NOT called by any other function (i.e., it's the entry
          // point)
          if (!topFunc && !calledFunctions.contains(func.getName()) &&
              !func.getBlocks().empty()) {
            topFunc = func;
          }
        }
      }

      if (topFunc) {
        // Found the top function - track its multi-dimensional array arguments
        for (auto arg : topFunc.getArguments()) {
          if (auto shapedType = llvm::dyn_cast<ShapedType>(arg.getType())) {
            if (shapedType.hasStaticShape() && shapedType.getRank() > 1) {
              state.topLevelFunctionArgs.insert(arg);
            }
          }
        }
      }
    }

    // Third pass: emit functions and non-stateful globals
    for (auto &op : *module.getBody()) {
      if (auto func = dyn_cast<func::FuncOp>(op)) {
        emitFunction(func);
        // Stop emission immediately if an error occurred
        if (state.encounteredError)
          return;
      } else if (auto cst = dyn_cast<memref::GlobalOp>(op)) {
        // Only emit non-stateful globals at module level
        // Stateful globals are emitted inside functions that use them
        if (!statefulGlobalNames.contains(cst.getSymName())) {
          emitGlobal(cst);
        }
      } else {
        emitError(&op, "is unsupported operation.");
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Entry of allo-translate
//===----------------------------------------------------------------------===//

LogicalResult allo::emitVivadoHLSWithFlag(ModuleOp module,
                                          llvm::raw_ostream &os,
                                          bool linearize_pointers) {
  AlloEmitterState state(os);
  state.linearize_pointers = linearize_pointers;
  hls::VhlsModuleEmitter(state).emitModule(module);
  return failure(state.encounteredError);
}

LogicalResult allo::emitVivadoHLS(ModuleOp module, llvm::raw_ostream &os) {
  return emitVivadoHLSWithFlag(module, os, false);
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
