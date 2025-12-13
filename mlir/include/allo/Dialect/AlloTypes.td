/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_TYPES
#define ALLO_TYPES

include "allo/Dialect/AlloDialect.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"

def Type : Allo_Type<"Type", [MemRefElementTypeInterface]> {
  let summary = "generic type";
  let mnemonic = "Type";
}

// // https://mlir.llvm.org/docs/Tutorials/DefiningAttributesAndTypes/#defining-custom-parsers-and-printers-using-assembly-formats
// https://github.com/llvm/torch-mlir/blob/main/include/torch-mlir/Dialect/Torch/IR/TorchTypes.td#L50
def LoopHandle : Allo_Type<"LoopHandle"> {
  let summary = "loop handle";
  let mnemonic = "LoopHandle";
}

def OpHandle : Allo_Type<"OpHandle"> {
  let summary = "op handle";
  let mnemonic = "OpHandle";
}

def Fixed : Allo_Type<"Fixed", [MemRefElementTypeInterface]> {
  let summary = "fixed point";
  let mnemonic = "Fixed";
  let parameters = (ins "std::size_t":$width, "std::size_t":$frac);
  let assemblyFormat = "`<` $width `,` $frac `>`";
}

def UFixed : Allo_Type<"UFixed", [MemRefElementTypeInterface]> {
  let summary = "unsigned fixed point";
  let mnemonic = "UFixed";
  let parameters = (ins "std::size_t":$width, "std::size_t":$frac);
  let assemblyFormat = "`<` $width `,` $frac `>`";
}

def Struct : Allo_Type<"Struct", [MemRefElementTypeInterface]> {
  let summary = "struct type";
  let mnemonic = "struct";
  let parameters = (ins ArrayRefParameter<"Type", "elementTypes">:$elementTypes);
  bit hasCustomAssemblyFormat = 1;
}

def Stream : Allo_Type<"Stream", [MemRefElementTypeInterface]> {
  let summary = "stream type";
  let mnemonic = "stream";
  let parameters = (ins AnyType:$baseType, "std::size_t":$depth);
  let assemblyFormat = "`<` $baseType `,` $depth `>`";
}

def StreamLike : TypeOrValueSemanticsContainer<Stream, "stream-like">;

#endif // ALLO_TYPES