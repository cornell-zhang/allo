/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_TRANSLATION_EMITVIVADOHLS_H
#define ALLO_TRANSLATION_EMITVIVADOHLS_H

#include "mlir/IR/BuiltinOps.h"
#include "allo/Translation/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "allo/Dialect/AlloOps.h"

namespace mlir {
namespace allo {

LogicalResult emitVivadoHLS(ModuleOp module, llvm::raw_ostream &os);
void registerEmitVivadoHLSTranslation();

// Vivado HLS ModuleEmitter class for inheritance
namespace vhls {
class ModuleEmitter : public AlloEmitterBase {
public:
  using operand_range = Operation::operand_range;
  explicit ModuleEmitter(AlloEmitterState &state) : AlloEmitterBase(state) {}

  /// SCF statement emitters.
  virtual void emitScfFor(scf::ForOp op);
  void emitScfIf(scf::IfOp op);
  void emitScfYield(scf::YieldOp op);

  /// Affine statement emitters.
  virtual void emitAffineFor(affine::AffineForOp op);
  void emitAffineIf(affine::AffineIfOp op);
  void emitAffineParallel(affine::AffineParallelOp op);
  void emitAffineApply(affine::AffineApplyOp op);
  template <typename OpType>
  void emitAffineMaxMin(OpType op, const char *syntax);
  void emitAffineLoad(affine::AffineLoadOp op);
  void emitAffineStore(affine::AffineStoreOp op);
  void emitAffineYield(affine::AffineYieldOp op);

  /// Memref-related statement emitters.
  template <typename OpType> void emitAlloc(OpType op);
  void emitLoad(memref::LoadOp op);
  void emitStore(memref::StoreOp op);
  void emitGetGlobal(memref::GetGlobalOp op);
  void emitGetGlobalFixed(allo::GetGlobalFixedOp op);
  void emitGlobal(memref::GlobalOp op);
  void emitSubView(memref::SubViewOp op);
  void emitReshape(memref::ReshapeOp op);

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
  void emitGeneralCast(mlir::UnrealizedConversionCastOp op);
  void emitGetBit(allo::GetIntBitOp op);
  void emitSetBit(allo::SetIntBitOp op);
  void emitGetSlice(allo::GetIntSliceOp op);
  void emitSetSlice(allo::SetIntSliceOp op);
  void emitBitReverse(allo::BitReverseOp op);
  void emitBitcast(arith::BitcastOp op);

  /// Stream operation emitters.
  virtual void emitStreamConstruct(allo::StreamConstructOp op);
  void emitStreamGet(allo::StreamGetOp op);
  void emitStreamPut(allo::StreamPutOp op);

  /// Top-level MLIR module emitter.
  virtual void emitModule(ModuleOp module);

protected:
  /// C++ component emitters.
  virtual void emitValue(Value val, unsigned rank = 0, bool isPtr = false,
                         std::string name = "");
  virtual void emitArrayDecl(Value array, bool isFunc = false, std::string name = "");
  unsigned emitNestedLoopHead(Value val);
  void emitNestedLoopTail(unsigned rank);
  void emitInfoAndNewLine(Operation *op);

  /// MLIR component and HLS C++ pragma emitters.
  void emitBlock(Block &block);
  virtual void emitLoopDirectives(Operation *op);
  virtual void emitArrayDirectives(Value memref);
  virtual void emitFunctionDirectives(func::FuncOp func, ArrayRef<Value> portList);
  virtual void emitFunction(func::FuncOp func);
  void emitHostFunction(func::FuncOp func);
};
} // namespace vhls

} // namespace allo
} // namespace mlir

#endif // ALLO_TRANSLATION_EMITVIVADOHLS_H