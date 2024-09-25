/*
 * Copyright HeteroCL authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HCL_MLIR_PASSES
#define HCL_MLIR_PASSES

include "mlir/Pass/PassBase.td"

def LoopTransformation : Pass<"loop-opt", "ModuleOp"> {
  let summary = "Loop transformation pass";
  let constructor = "mlir::hcl::createLoopTransformationPass()";
}

def DataPlacement : Pass<"data-placement", "ModuleOp"> {
  let summary = "Data placement pass";
  let constructor = "mlir::hcl::createDataPlacementPass()";
}

def AnyWidthInteger : Pass<"anywidth-integer", "ModuleOp"> {
  let summary = "Transform anywidth-integer input to 64-bit";
  let constructor = "mlir::hcl::createAnyWidthIntegerPass()";
}

def MoveReturnToInput : Pass<"return-to-input", "ModuleOp"> {
  let summary = "Move return values to input argument list";
  let constructor = "mlir::hcl::createMoveReturnToInputPass()";
}

def LegalizeCast : Pass<"legalize-cast", "ModuleOp"> {
  let summary = "Legalize cast operations";
  let constructor = "mlir::hcl::createLegalizeCastPass()";
}

def RemoveStrideMap : Pass<"remove-stride-map", "ModuleOp"> {
  let summary = "Remove stride map from partitioned memref";
  let constructor = "mlir::hcl::createRemoveStrideMapPass()";
}

def MemRefDCE : Pass<"memref-dce", "ModuleOp"> {
  let summary = "Remove MemRefs that are never loaded from";
  let constructor = "mlir::hcl::createMemRefDCEPass()";
}

#endif // HCL_MLIR_PASSES