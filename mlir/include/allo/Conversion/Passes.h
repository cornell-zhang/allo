/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_CONVERSION_PASSES_H
#define ALLO_CONVERSION_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

#include "allo/Dialect/AlloOps.h"

namespace mlir {
namespace allo {

// Allo Dialect -> LLVM Dialect
std::unique_ptr<OperationPass<ModuleOp>> createAlloToLLVMLoweringPass();
std::unique_ptr<OperationPass<ModuleOp>> createFixedPointToIntegerPass();
std::unique_ptr<OperationPass<ModuleOp>> createLowerCompositeTypePass();
std::unique_ptr<OperationPass<ModuleOp>> createLowerBitOpsPass();
std::unique_ptr<OperationPass<ModuleOp>> createLowerTransformLayoutOpsPass();
std::unique_ptr<OperationPass<ModuleOp>> createLowerPrintOpsPass();

bool applyAlloToLLVMLoweringPass(ModuleOp &module, MLIRContext &context);
bool applyFixedPointToInteger(ModuleOp &module);
bool applyLowerCompositeType(ModuleOp &module);
bool applyLowerBitOps(ModuleOp &module);
bool applyLowerTransformLayoutOps(ModuleOp &module);
bool applyLowerPrintOps(ModuleOp &module);

/// Registers all Allo conversion passes
void registerAlloConversionPasses();

#define GEN_PASS_DECL
#include "allo/Conversion/Passes.h.inc"

} // namespace allo
} // namespace mlir

#endif // ALLO_CONVERSION_PASSES_H