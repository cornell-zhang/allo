/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_TRANSLATION_EMITINTELHLS_H
#define ALLO_TRANSLATION_EMITINTELHLS_H

#include "allo/Translation/EmitBaseHLS.h"

namespace mlir {
namespace allo {
namespace hls {

/// Intel HLS (DPC++/SYCL) ModuleEmitter
class IntelModuleEmitter : public ModuleEmitterBase {
public:
  using operand_range = Operation::operand_range;
  explicit IntelModuleEmitter(AlloEmitterState &state) : ModuleEmitterBase(state) {}

  /// Affine statement emitters.
  void emitAffineFor(affine::AffineForOp op) override;
  void emitAffineLoad(affine::AffineLoadOp op) override;
  void emitAffineStore(affine::AffineStoreOp op) override;
  void emitAffineApply(affine::AffineApplyOp op) override;
  void emitAffineYield(affine::AffineYieldOp op) override;

  /// Memref-related statement emitters.
  template <typename OpType> void emitAlloc(OpType op);
  void emitLoad(memref::LoadOp op) override;
  void emitStore(memref::StoreOp op) override;

  /// Special operation emitters.
  void emitConstant(arith::ConstantOp op) override;
  template <typename CastOpType> void emitCast(CastOpType op);

  /// Top-level MLIR module emitter.
  void emitModule(ModuleOp module) override;

protected:
  /// C++ component emitters.
  void emitValue(Value val, unsigned rank = 0, bool isPtr = false,
                 std::string name = "") override {
    emitValueImpl(val, rank, isPtr, name, false);
  }
  void emitValueImpl(Value val, unsigned rank, bool isPtr,
                     std::string name, bool noType);

  void emitArrayDecl(Value array, bool isFunc = false, std::string name = "");
  void emitBufferDecl(Value array, bool isAccessor = false,
                      bool isReadOnly = false, std::string name = "") override;
  unsigned emitNestedLoopHead(Value val) override;

  /// MLIR component and HLS C++ pragma emitters.
  void emitBlock(Block &block) override;
  void emitLoopDirectives(Operation *op) override;
  void emitFunction(func::FuncOp func, bool isAccessor) override;
};

} // namespace hls

LogicalResult emitIntelHLS(ModuleOp module, llvm::raw_ostream &os);
void registerEmitIntelHLSTranslation();

} // namespace allo
} // namespace mlir

#endif // ALLO_TRANSLATION_EMITINTELHLS_H