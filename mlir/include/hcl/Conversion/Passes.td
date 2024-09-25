/*
 * Copyright HeteroCL authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HCL_MLIR_PASSES
#define HCL_MLIR_PASSES

include "mlir/Pass/PassBase.td"

def HCLToLLVMLowering : Pass<"hcl-lower-to-llvm", "ModuleOp"> {
  let summary = "HCL to LLVM conversion pass";
  let constructor = "mlir::hcl::createHCLToLLVMLoweringPass()";
}

def FixedToInteger : Pass<"fixed-to-integer", "ModuleOp"> {
  let summary = "Fixed-point operations to integer";
  let constructor = "mlir::hcl::createFixedPointToIntegerPass()";
}

def LowerCompositeType : Pass<"lower-composite-type", "ModuleOp"> {
  let summary = "Lower composite types";
  let constructor = "mlir::hcl::createLowerCompositeTypePass()";
}

def LowerBitOps : Pass<"lower-bit-ops", "ModuleOp"> {
  let summary = "Lower bit operations";
  let constructor = "mlir::hcl::createLowerBitOpsPass()";
}

def LowerPrintOps : Pass<"lower-print-ops", "ModuleOp"> {
  let summary = "Lower print operations";
  let constructor = "mlir::hcl::createLowerPrintOpsPass()";
}

#endif // HCL_MLIR_PASSES