/*
 * Copyright HeteroCL authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HCL_TRANSFORMS_PASSES_H
#define HCL_TRANSFORMS_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace hcl {

std::unique_ptr<OperationPass<ModuleOp>> createLoopTransformationPass();
std::unique_ptr<OperationPass<ModuleOp>> createAnyWidthIntegerPass();
std::unique_ptr<OperationPass<ModuleOp>> createMoveReturnToInputPass();
std::unique_ptr<OperationPass<ModuleOp>> createLegalizeCastPass();
std::unique_ptr<OperationPass<ModuleOp>> createRemoveStrideMapPass();
std::unique_ptr<OperationPass<ModuleOp>> createMemRefDCEPass();
std::unique_ptr<OperationPass<ModuleOp>> createDataPlacementPass();

bool applyLoopTransformation(ModuleOp &f);
bool applyAnyWidthInteger(ModuleOp &module);
bool applyMoveReturnToInput(ModuleOp &module);
bool applyLegalizeCast(ModuleOp &module);
bool applyRemoveStrideMap(ModuleOp &module);
bool applyMemRefDCE(ModuleOp &module);
bool applyDataPlacement(ModuleOp &module);

/// Registers all HCL transformation passes
void registerHCLPasses();

} // namespace hcl
} // namespace mlir

#endif // HCL_TRANSFORMS_PASSES_H