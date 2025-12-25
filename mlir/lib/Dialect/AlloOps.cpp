/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//===- AlloOps.cpp - allo dialect ops ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "allo/Dialect/AlloOps.h"
#include "allo/Dialect/AlloAttrs.h"
#include "allo/Dialect/AlloDialect.h"
#include "allo/Dialect/AlloTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/FunctionImplementation.h"

namespace mlir {
namespace allo {

ParseResult CustomizationOp::parse(OpAsmParser &parser,
                                   OperationState &result) {
  auto buildFuncType =
      [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
         function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };

  return function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      getFunctionTypeAttrName(result.name), buildFuncType, nullptr, nullptr);
  // getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void CustomizationOp::print(OpAsmPrinter &p) {
  function_interface_impl::printFunctionOp(p, *this, /*isVariadic=*/false,
                                           getFunctionTypeAttrName(), nullptr,
                                           nullptr);
}

LogicalResult CustomizationOp::verify() {
  // If this function is external there is nothing to do.
  if (isExternal())
    return success();

  // Verify that the argument list of the function and the arg list of the entry
  // block line up.  The trait already verified that the number of arguments is
  // the same between the signature and the block.
  auto fnInputTypes = getFunctionType().getInputs();
  Block &entryBlock = front();
  for (unsigned i = 0, e = entryBlock.getNumArguments(); i != e; ++i)
    if (fnInputTypes[i] != entryBlock.getArgument(i).getType())
      return emitOpError("type of entry block argument #")
             << i << '(' << entryBlock.getArgument(i).getType()
             << ") must match the type of the corresponding argument in "
             << "function signature(" << fnInputTypes[i] << ')';

  return success();
}

//===----------------------------------------------------------------------===//
// General helpers for comparison ops
//===----------------------------------------------------------------------===//

// Return the type of the same shape (scalar, vector or tensor) containing i1.
static Type getI1SameShape(Type type) {
  auto i1Type = IntegerType::get(type.getContext(), 1);
  if (auto tensorType = llvm::dyn_cast<RankedTensorType>(type))
    return RankedTensorType::get(tensorType.getShape(), i1Type);
  if (llvm::isa<UnrankedTensorType>(type))
    return UnrankedTensorType::get(i1Type);
  if (auto vectorType = llvm::dyn_cast<VectorType>(type))
    return VectorType::get(vectorType.getShape(), i1Type);
  return i1Type;
}

//===----------------------------------------------------------------------===//
// CmpFixedOp
//===----------------------------------------------------------------------===//

static void buildCmpFixedOp(OpBuilder &build, OperationState &result,
                            CmpFixedPredicate predicate, Value lhs, Value rhs) {
  result.addOperands({lhs, rhs});
  result.types.push_back(getI1SameShape(lhs.getType()));
  // FIXME: cannot call member function ‘mlir::StringAttr
  // mlir::allo::CmpFixedOp::getPredicateAttrName()’ without object
  // result.addAttribute(CmpFixedOp::getPredicateAttrName(),
  //                     build.getI64IntegerAttr(static_cast<int64_t>(predicate)));
}

} // namespace allo
} // namespace mlir

//===----------------------------------------------------------------------===//
// Custom build method implementations
//===----------------------------------------------------------------------===//

void mlir::allo::PipelineOp::build(OpBuilder &builder, OperationState &state,
                                   allo::LoopHandleType loop, uint64_t ii) {
  state.addTypes(loop);
  state.addAttribute("ii", builder.getUI32IntegerAttr(ii));
}

void mlir::allo::ThreadBindOp::build(OpBuilder &builder, OperationState &state,
                                     allo::LoopHandleType loop, uint64_t dim) {
  state.addAttribute("dim", allo::NDRangeDimKindEnumAttr::get(
                                builder.getContext(),
                                static_cast<allo::NDRangeDimKindEnum>(dim)));
}

void mlir::allo::ApplyOp::build(OpBuilder &builder, OperationState &state,
                                StringRef callee, ArrayRef<Value> operands) {
  state.addOperands(operands);
  state.addAttribute("callee",
                     SymbolRefAttr::get(builder.getContext(), callee));
}

void mlir::allo::StructGetOp::build(OpBuilder &builder, OperationState &state,
                                    Value input, size_t index) {
  auto structType = llvm::cast<allo::StructType>(input.getType());
  Type resultType = structType.getElementTypes()[index];
  state.addOperands(input);
  state.addAttribute("index", builder.getI64IntegerAttr(index));
  state.addTypes(resultType);
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "allo/Dialect/AlloOps.cpp.inc"
