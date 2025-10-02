/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_MLIR_PASSES
#define ALLO_MLIR_PASSES

include "mlir/Pass/PassBase.td"

def AlloToLLVMLowering : Pass<"allo-lower-to-llvm", "ModuleOp"> {
  let summary = "Allo to LLVM conversion pass";
  let constructor = "mlir::allo::createAlloToLLVMLoweringPass()";
}

def FixedToInteger : Pass<"fixed-to-integer", "ModuleOp"> {
  let summary = "Fixed-point operations to integer";
  let constructor = "mlir::allo::createFixedPointToIntegerPass()";
}

def LowerCompositeType : Pass<"lower-composite-type", "ModuleOp"> {
  let summary = "Lower composite types";
  let constructor = "mlir::allo::createLowerCompositeTypePass()";
}

def LowerBitOps : Pass<"lower-bit-ops", "ModuleOp"> {
  let summary = "Lower bit operations";
  let constructor = "mlir::allo::createLowerBitOpsPass()";
}

def LowerTransformLayoutOps : Pass<"lower-transform-layout-ops", "ModuleOp"> {
  let summary = "Lower transform-layout operations";
  let constructor = "mlir::allo::createLowerTransformLayoutOpsPass()";
}

def LowerStoreSliceOps : Pass<"lower-store-slice-ops", "ModuleOp"> {
  let summary = "Lower store-slice operations";
  let constructor = "mlir::allo::createLowerStoreSliceOpsPass()";
}

def LowerLoadSliceOps : Pass<"lower-load-slice-ops", "ModuleOp"> {
  let summary = "Lower load-slice operations";
  let constructor = "mlir::allo::createLowerLoadSliceOpsPass()";
}

def LowerPrintOps : Pass<"lower-print-ops", "ModuleOp"> {
  let summary = "Lower print operations";
  let constructor = "mlir::allo::createLowerPrintOpsPass()";
}

#endif // ALLO_MLIR_PASSES