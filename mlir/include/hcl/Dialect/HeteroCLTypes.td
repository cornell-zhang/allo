/*
 * Copyright HeteroCL authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HCL_TYPES
#define HCL_TYPES

include "hcl/Dialect/HeteroCLDialect.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

def Type : HeteroCL_Type<"Type", [MemRefElementTypeInterface]> {
  let summary = "generic type";
  let mnemonic = "Type";
}

// // https://mlir.llvm.org/docs/Tutorials/DefiningAttributesAndTypes/#defining-custom-parsers-and-printers-using-assembly-formats
// https://github.com/llvm/torch-mlir/blob/main/include/torch-mlir/Dialect/Torch/IR/TorchTypes.td#L50
def LoopHandle : HeteroCL_Type<"LoopHandle"> {
  let summary = "loop handle";
  let mnemonic = "LoopHandle";
}

def OpHandle : HeteroCL_Type<"OpHandle"> {
  let summary = "op handle";
  let mnemonic = "OpHandle";
}

def Fixed : HeteroCL_Type<"Fixed", [MemRefElementTypeInterface]> {
  let summary = "fixed point";
  let mnemonic = "Fixed";
  let parameters = (ins "std::size_t":$width, "std::size_t":$frac);
  let assemblyFormat = "`<` $width `,` $frac `>`";
}

def UFixed : HeteroCL_Type<"UFixed", [MemRefElementTypeInterface]> {
  let summary = "unsigned fixed point";
  let mnemonic = "UFixed";
  let parameters = (ins "std::size_t":$width, "std::size_t":$frac);
  let assemblyFormat = "`<` $width `,` $frac `>`";
}

def Struct : HeteroCL_Type<"Struct", [MemRefElementTypeInterface]> {
  let summary = "struct type";
  let mnemonic = "struct";
  let parameters = (ins ArrayRefParameter<"Type", "elementTypes">:$elementTypes);
  bit hasCustomAssemblyFormat = 1;
}

#endif // HCL_TYPES