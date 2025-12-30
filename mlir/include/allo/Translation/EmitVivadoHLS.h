/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_TRANSLATION_EMITVIVADOHLS_H
#define ALLO_TRANSLATION_EMITVIVADOHLS_H

#include "allo/Translation/EmitBaseHLS.h"

namespace mlir {
namespace allo {
namespace hls {

/// Vivado HLS ModuleEmitter (can be inherited by XLS)
class VhlsModuleEmitter : public ModuleEmitterBase {
public:
  using operand_range = Operation::operand_range;
  explicit VhlsModuleEmitter(AlloEmitterState &state) : ModuleEmitterBase(state) {}

  /// SCF statement emitters.
  void emitScfFor(scf::ForOp op) override;
  void emitScfIf(scf::IfOp op) override;
  void emitScfWhile(scf::WhileOp op) override;
  void emitScfCondition(scf::ConditionOp op) override;
  void emitScfYield(scf::YieldOp op) override;

  /// Affine statement emitters.
  void emitAffineFor(affine::AffineForOp op) override;
  void emitAffineIf(affine::AffineIfOp op) override;
  void emitAffineParallel(affine::AffineParallelOp op) override;
  void emitAffineApply(affine::AffineApplyOp op) override;
  template <typename OpType>
  void emitAffineMaxMin(OpType op, const char *syntax);
  void emitAffineLoad(affine::AffineLoadOp op) override;
  void emitAffineStore(affine::AffineStoreOp op) override;
  void emitAffineYield(affine::AffineYieldOp op) override;

  /// Memref-related statement emitters.
  template <typename OpType> void emitAlloc(OpType op);
  void emitLoad(memref::LoadOp op) override;
  void emitStore(memref::StoreOp op) override;
  void emitGetGlobal(memref::GetGlobalOp op) override;
  void emitGetGlobalFixed(allo::GetGlobalFixedOp op) override;
  void emitGlobal(memref::GlobalOp op) override;
  void emitSubView(memref::SubViewOp op) override;
  void emitReshape(memref::ReshapeOp op) override;

  /// Tensor-related statement emitters.
  void emitTensorExtract(tensor::ExtractOp op) override;
  void emitTensorInsert(tensor::InsertOp op) override;
  void emitDim(memref::DimOp op) override;
  void emitRank(memref::RankOp op) override;

  /// Special operation emitters.
  void emitCall(func::CallOp op) override;
  void emitSelect(arith::SelectOp op) override;
  void emitConstant(arith::ConstantOp op) override;
  template <typename CastOpType> void emitCast(CastOpType op);
  void emitGeneralCast(mlir::UnrealizedConversionCastOp op) override;
  void emitGetBit(allo::GetIntBitOp op) override;
  void emitSetBit(allo::SetIntBitOp op) override;
  void emitGetSlice(allo::GetIntSliceOp op) override;
  void emitSetSlice(allo::SetIntSliceOp op) override;
  void emitBitReverse(allo::BitReverseOp op) override;
  void emitBitcast(arith::BitcastOp op) override;

  /// Stream operation emitters.
  void emitStreamConstruct(allo::StreamConstructOp op) override;
  void emitStreamGet(allo::StreamGetOp op) override;
  void emitStreamPut(allo::StreamPutOp op) override;

  /// Top-level MLIR module emitter.
  void emitModule(ModuleOp module) override;

protected:
  /// C++ component emitters.
  void emitValue(Value val, unsigned rank = 0, bool isPtr = false,
                 std::string name = "") override;
  void emitArrayDecl(Value array, bool isFunc = false, std::string name = "") override;
  unsigned emitNestedLoopHead(Value val) override;

  /// MLIR component and HLS C++ pragma emitters.
  void emitBlock(Block &block) override;
  void emitLoopDirectives(Operation *op) override;
  void emitArrayDirectives(Value memref) override;
  void emitFunctionDirectives(func::FuncOp func, ArrayRef<Value> portList) override;
  void emitFunction(func::FuncOp func) override;
  void emitHostFunction(func::FuncOp func) override;
};

} // namespace hls

LogicalResult emitVivadoHLS(ModuleOp module, llvm::raw_ostream &os);
LogicalResult emitVivadoHLSWithFlag(ModuleOp module, llvm::raw_ostream &os, bool linearize_pointers);
void registerEmitVivadoHLSTranslation();

} // namespace allo
} // namespace mlir

#endif // ALLO_TRANSLATION_EMITVIVADOHLS_H