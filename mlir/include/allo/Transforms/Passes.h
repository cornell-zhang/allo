/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_TRANSFORMS_PASSES_H
#define ALLO_TRANSFORMS_PASSES_H

#include "mlir/CAPI/IR.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace allo {

std::unique_ptr<OperationPass<ModuleOp>> createLoopTransformationPass();
std::unique_ptr<OperationPass<ModuleOp>> createAnyWidthIntegerPass();
std::unique_ptr<OperationPass<ModuleOp>> createMoveReturnToInputPass();
std::unique_ptr<OperationPass<ModuleOp>> createLegalizeCastPass();
std::unique_ptr<OperationPass<ModuleOp>> createRemoveStrideMapPass();
std::unique_ptr<OperationPass<ModuleOp>> createMemRefDCEPass();
std::unique_ptr<OperationPass<ModuleOp>> createDataPlacementPass();
std::unique_ptr<OperationPass<ModuleOp>> createCopyOnWritePass();

bool applyLoopTransformation(ModuleOp &f);
bool applyAnyWidthInteger(ModuleOp &module);
bool applyMoveReturnToInput(ModuleOp &module);
bool applyLegalizeCast(ModuleOp &module);
bool applyRemoveStrideMap(ModuleOp &module);
bool applyMemRefDCE(ModuleOp &module);
bool applyDataPlacement(ModuleOp &module);
bool applyCopyOnWrite(ModuleOp &module);
void applyCopyOnWriteOnFunction(Operation &func);
ModuleOp applyUnifyKernels(ModuleOp &module1, ModuleOp &module2, int loop_num);

/// Registers all Allo transformation passes
void registerAlloPasses();

} // namespace allo
} // namespace mlir

#endif // ALLO_TRANSFORMS_PASSES_H