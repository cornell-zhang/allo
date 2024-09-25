/*
 * Copyright HeteroCL authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HCL_CONVERSION_PASSES_H
#define HCL_CONVERSION_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

#include "hcl/Dialect/HeteroCLOps.h"

namespace mlir {
namespace hcl {

// HeteroCL Dialect -> LLVM Dialect
std::unique_ptr<OperationPass<ModuleOp>> createHCLToLLVMLoweringPass();
std::unique_ptr<OperationPass<ModuleOp>> createFixedPointToIntegerPass();
std::unique_ptr<OperationPass<ModuleOp>> createLowerCompositeTypePass();
std::unique_ptr<OperationPass<ModuleOp>> createLowerBitOpsPass();
std::unique_ptr<OperationPass<ModuleOp>> createLowerPrintOpsPass();

bool applyHCLToLLVMLoweringPass(ModuleOp &module, MLIRContext &context);
bool applyFixedPointToInteger(ModuleOp &module);
bool applyLowerCompositeType(ModuleOp &module);
bool applyLowerBitOps(ModuleOp &module);
bool applyLowerPrintOps(ModuleOp &module);

/// Registers all HCL conversion passes
void registerHCLConversionPasses();

#define GEN_PASS_CLASSES
#include "hcl/Conversion/Passes.h.inc"

} // namespace hcl
} // namespace mlir

#endif // HCL_CONVERSION_PASSES_H