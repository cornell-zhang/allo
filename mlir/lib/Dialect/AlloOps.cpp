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
#include "mlir/IR/Matchers.h"
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

//===----------------------------------------------------------------------===//
// MetaIfOp
//===----------------------------------------------------------------------===//

void buildTerminatedBody(OpBuilder &builder, Location loc) {
  YieldOp::create(builder, loc);
}

void MetaIfOp::build(OpBuilder &builder, OperationState &result, Value cond,
                     bool withElseRegion) {
  result.addOperands(cond);
  Region *thenRegion = result.addRegion();
  Region *elseRegion = result.addRegion();
  MetaIfOp::ensureTerminator(*thenRegion, builder, result.location);
  if (withElseRegion)
    MetaIfOp::ensureTerminator(*elseRegion, builder, result.location);
}

void MetaIfOp::build(OpBuilder &builder, OperationState &result, Value cond,
                     function_ref<void(OpBuilder &, Location)> thenBuilder,
                     function_ref<void(OpBuilder &, Location)> elseBuilder) {
  result.addOperands(cond);
  Region *thenRegion = result.addRegion();
  Region *elseRegion = result.addRegion();

  OpBuilder::InsertionGuard guard(builder);
  builder.createBlock(thenRegion);
  thenBuilder(builder, result.location);

  if (elseBuilder) {
    builder.createBlock(elseRegion);
    elseBuilder(builder, result.location);
  }
}

ParseResult MetaIfOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parse the condition.
  OpAsmParser::UnresolvedOperand cond;
  if (parser.parseOperand(cond) ||
      parser.resolveOperand(cond, parser.getBuilder().getI1Type(),
                            result.operands))
    return failure();

  // Parse the 'then' region.
  Region *thenRegion = result.addRegion();
  if (parser.parseRegion(*thenRegion, /*arguments=*/{}, /*argTypes=*/{}))
    return failure();
  MetaIfOp::ensureTerminator(*thenRegion, parser.getBuilder(), result.location);

  // Parse the 'else' region.
  Region *elseRegion = result.addRegion();
  if (succeeded(parser.parseOptionalKeyword("else"))) {
    if (parser.parseRegion(*elseRegion, /*arguments=*/{}, /*argTypes=*/{}))
      return failure();
    MetaIfOp::ensureTerminator(*elseRegion, parser.getBuilder(),
                               result.location);
  }

  // Parse the optional attribute dictionary.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  return success();
}

void MetaIfOp::print(OpAsmPrinter &p) {
  p << " " << getCondition();
  p << " ";
  p.printRegion(getThenRegion(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);

  if (!getElseRegion().empty()) {
    p << " else ";
    p.printRegion(getElseRegion(),
                  /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/true);
  }

  p.printOptionalAttrDict(getOperation()->getAttrs());
}

LogicalResult MetaIfOp::verify() {
  if (getNumRegions() != 2)
    return emitOpError("expected 2 regions");
  if (!getThenRegion().hasOneBlock())
    return emitOpError("expected exactly 1 block in the 'then' region");
  if (!getElseRegion().empty() && !getElseRegion().hasOneBlock())
    return emitOpError("expected at most 1 block in the 'else' region");
  return success();
}

Block *MetaIfOp::thenBlock() { return &getThenRegion().back(); }
YieldOp MetaIfOp::thenYield() {
  return cast<YieldOp>(thenBlock()->getTerminator());
}
Block *MetaIfOp::elseBlock() {
  Region &r = getElseRegion();
  if (r.empty())
    return nullptr;
  return &r.back();
}
YieldOp MetaIfOp::elseYield() {
  return cast<YieldOp>(elseBlock()->getTerminator());
}

void MetaIfOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
    // TODO
}



//===----------------------------------------------------------------------===//
// MetaForOp
//===----------------------------------------------------------------------===//

void MetaForOp::build(OpBuilder &builder, OperationState &result,
                      Value lowerBound, Value upperBound, Value step,
                      ValueRange initArgs,
                      BodyBuilderFn bodyBuilder) {
  result.addOperands({lowerBound, upperBound, step});

  Region *bodyRegion = result.addRegion();
  Block *bodyBlock = builder.createBlock(bodyRegion);
  // Add induction variable
  bodyBlock->addArgument(builder.getIndexType(), result.location);

  if (bodyBuilder) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(bodyBlock);
    bodyBuilder(builder, result.location, bodyBlock->getArgument(0), {});
  }
  MetaForOp::ensureTerminator(*bodyRegion, builder, result.location);
}

ParseResult MetaForOp::parse(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  OpAsmParser::Argument inductionVariable;
  OpAsmParser::UnresolvedOperand lb, ub, step;

  // Parse the induction variable followed by '='.
  if (parser.parseArgument(inductionVariable) || parser.parseEqual() ||
      parser.parseOperand(lb) || parser.parseKeyword("to") ||
      parser.parseOperand(ub) || parser.parseKeyword("step") ||
      parser.parseOperand(step))
    return failure();

  // Parse the optional attribute dictionary.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  // Parse the body region.
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, inductionVariable))
    return failure();

  MetaForOp::ensureTerminator(*body, builder, result.location);

  // Resolve operands.
  if (parser.resolveOperand(lb, builder.getIndexType(), result.operands) ||
      parser.resolveOperand(ub, builder.getIndexType(), result.operands) ||
      parser.resolveOperand(step, builder.getIndexType(), result.operands))
    return failure();

  return success();
}

void MetaForOp::print(OpAsmPrinter &p) {
  p << " " << getInductionVar() << " = " << getLowerBound() << " to "
    << getUpperBound() << " step " << getStep();
  p.printOptionalAttrDict(getOperation()->getAttrs());
  p << " ";
  p.printRegion(getRegion(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);
}

LogicalResult MetaForOp::verify() {
  IntegerAttr lb;
  if (!matchPattern(getLowerBound(), m_Constant(&lb)))
    return emitOpError("lower bound must be constant");
  
  IntegerAttr ub;
  if (!matchPattern(getUpperBound(), m_Constant(&ub)))
    return emitOpError("upper bound must be constant");

  IntegerAttr step;
  if (!matchPattern(getStep(), m_Constant(&step)))
    return emitOpError("step must be constant");

  if (step.getValue().isZero())
    return emitOpError("constant step operand must be non-zero");

  return success();
}

LogicalResult MetaForOp::verifyRegions() {
  if (getRegion().empty())
    return emitOpError("region must not be empty");
  if (getRegion().front().getNumArguments() != 1)
    return emitOpError("region must have exactly one argument");
  return success();
}

std::optional<APInt> MetaForOp::getConstantStep() {
  IntegerAttr step;
  if (matchPattern(getStep(), m_Constant(&step)))
    return step.getValue();
  return std::nullopt;
}

void MetaForOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
    // TODO
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
