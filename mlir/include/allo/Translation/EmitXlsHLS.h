/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_TRANSLATION_EMITXLSHLS_H
#define ALLO_TRANSLATION_EMITXLSHLS_H

#include "allo/Translation/EmitVivadoHLS.h"

namespace mlir {
namespace allo {
namespace hls {

/// XLS HLS ModuleEmitter (inherits from VhlsModuleEmitter)
class XlsModuleEmitter : public VhlsModuleEmitter {
public:
  using operand_range = Operation::operand_range;
  explicit XlsModuleEmitter(AlloEmitterState &state)
      : VhlsModuleEmitter(state) {}

  /// Top-level MLIR module emitter.
  void emitModule(ModuleOp module) override;

  /// Override methods that need XLS-specific behavior.
  void emitFunctionDirectives(func::FuncOp func,
                              ArrayRef<Value> portList) override;
  void emitArrayDecl(Value array, bool isFunc = false,
                     std::string name = "") override;
  void emitLoopDirectives(Operation *op) override;
  void emitStreamConstruct(allo::StreamConstructOp op) override;
  void emitArrayDirectives(Value memref) override;
  void emitFunction(func::FuncOp func) override;

  /// Affine statement emitters.
  void emitAffineFor(affine::AffineForOp op) override;

  /// SCF statement emitters.
  void emitScfFor(scf::ForOp op) override;

  /// Memref-related statement emitters.
  void emitLoad(memref::LoadOp op) override;
  void emitStore(memref::StoreOp op) override;
  void emitAffineLoad(affine::AffineLoadOp op) override;
  void emitAffineStore(affine::AffineStoreOp op) override;

protected:
  /// C++ component emitters.
  void emitValue(Value val, unsigned rank = 0, bool isPtr = false,
                 std::string name = "") override;

  /// Helper to emit flattened index for multi-dimensional array access.
  void emitFlattenedIndex(MemRefType memrefType,
                          Operation::operand_range indices);

  /// Helper method to get XLS-specific type names.
  SmallString<16> getTypeName(Type valType);
  SmallString<16> getTypeName(Value val);

  /// Track array arguments that should become local arrays populated from
  /// channels.
  bool inArrayFunction = false;
  DenseMap<Value, std::string> arrayArgToLocalName;
  DenseMap<Value, std::string> arrayArgToChannelName;
  Value returnArray = nullptr;
  std::string outputChannelName = "out";
};

} // namespace hls

LogicalResult emitXlsHLS(ModuleOp module, llvm::raw_ostream &os,
                         bool useMemory = false);
void registerEmitXlsHLSTranslation();

} // namespace allo
} // namespace mlir

#endif // ALLO_TRANSLATION_EMITXLSHLS_H
