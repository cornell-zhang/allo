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
#include "mlir/IR/SymbolTable.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

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
// StreamGlobalOp
//===----------------------------------------------------------------------===//

LogicalResult StreamGlobalOp::verify() {
  Type type = getElementType();
  if (!llvm::isa<StreamType>(type)) {
     return emitOpError("element type of global stream must be stream type");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// GlobalStreamGetOp
//===----------------------------------------------------------------------===//

LogicalResult GlobalStreamGetOp::verify() {
  Operation *symbol = SymbolTable::lookupNearestSymbolFrom(*this, getGlobalAttr());
  auto global = llvm::dyn_cast_or_null<StreamGlobalOp>(symbol);
  if (!global)
    return emitOpError("global stream not found: ") << getGlobal();
  
  auto type = global.getElementType();
  auto streamType = llvm::cast<StreamType>(type);
  if (getResult().getType() != streamType.getBaseType())
    return emitOpError("result type mismatch");
  
  AffineMap map = getMap();

  if (map.getNumDims() + map.getNumSymbols() != getIndices().size())
    return emitOpError("affine map dim & symbol count mismatch");

  int64_t rank = global.getShape().size();
  if (map.getNumResults() != rank)
    return emitOpError("affine map result count mismatch: expected ")
           << rank << ", got " << map.getNumResults();

  for (Value idx : getIndices())
    if (!idx.getType().isIndex())
      return emitOpError("indices must be of index type");

  return success();
}

void GlobalStreamGetOp::print(OpAsmPrinter &p) {
  p << " @" << getGlobal() << '[';
  if (AffineMapAttr mapAttr =
          (*this)->getAttrOfType<AffineMapAttr>("map")) {
    p.printAffineMapOfSSAIds(mapAttr, getIndices());
  }
  p << ']';
  p.printOptionalAttrDict((*this)->getAttrs(),{"map", "global"});
  p << " : " << getResult().getType();
}


ParseResult GlobalStreamGetOp::parse(OpAsmParser &parser,
                                     OperationState &result) {
  auto &builder = parser.getBuilder();
  auto indexTy = builder.getIndexType();

  FlatSymbolRefAttr globalAttr;
  if (parser.parseAttribute(globalAttr, "global", result.attributes))
    return failure();

  AffineMapAttr mapAttr;
  SmallVector<OpAsmParser::UnresolvedOperand, 2> mapOperands;
  if (parser.parseAffineMapOfSSAIds(mapOperands, mapAttr, "map", result.attributes))
    return failure();

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  Type resultType;
  if (parser.parseColonType(resultType))
    return failure();

  result.addTypes(resultType);

  if (parser.resolveOperands(mapOperands, indexTy, result.operands))
    return failure();

  return success();
}

template <typename OpTy>
static void simplifyStreamAffineMap(OpTy op) {
  AffineMap map = op.getMap();
  SmallVector<AffineExpr, 4> dimReplacements;
  SmallVector<AffineExpr, 4> symReplacements;

  auto *context = map.getContext();
  unsigned numDims = map.getNumDims();
  unsigned numSymbols = map.getNumSymbols();
  auto indices = op.getIndices(); 

  // Collect dim operands (these must stay)
  SmallVector<Value, 4> newOperands;
  for (unsigned i = 0; i < numDims; ++i) {
    Value opValue = indices[i];
    if (auto constOp = opValue.template getDefiningOp<arith::ConstantOp>()) {
      int64_t val = llvm::cast<IntegerAttr>(constOp.getValue()).getInt();
      dimReplacements.push_back(getAffineConstantExpr(val, context));
    } else {
      dimReplacements.push_back(getAffineDimExpr(newOperands.size(), context));
      newOperands.push_back(opValue);
    }
    // newOperands.push_back(indices[i]);
  }
  auto newDims = newOperands.size();
  if (indices.size() > numDims + numSymbols) {
    newOperands.push_back(indices[numDims + numSymbols]);
  }
  // Replace symbols with constants
  for (unsigned i = 0; i < numSymbols; ++i) {
    Value opValue = indices[numDims + i];
    if (auto constOp = opValue.template getDefiningOp<arith::ConstantOp>()) {
      int64_t val = llvm::cast<IntegerAttr>(constOp.getValue()).getInt();
      symReplacements.push_back(getAffineConstantExpr(val, context));
    } else {
      op.emitOpError() << "expected arith.constant for symbol, but got: " << opValue;
      return;
    }
  }

  // Replace only symbols
  AffineMap newMap = map.replaceDimsAndSymbols(dimReplacements, symReplacements, newDims, 0);
  newMap = mlir::compressUnusedSymbols(mlir::simplifyAffineMap(newMap));
  op.setMapAttr(AffineMapAttr::get(newMap));
  // Update operands: keep dims only
  op.getIndicesMutable().assign(newOperands);
}

void GlobalStreamGetOp::simplifyAffineMap() {
  simplifyStreamAffineMap(*this);
}

//===----------------------------------------------------------------------===//
// GlobalStreamPutOp
//===----------------------------------------------------------------------===//

LogicalResult GlobalStreamPutOp::verify() {
  Operation *symbol = SymbolTable::lookupNearestSymbolFrom(*this, getGlobalAttr());
  auto global = llvm::dyn_cast_or_null<StreamGlobalOp>(symbol);
  if (!global)
    return emitOpError("global stream not found: ") << getGlobal();
  
  auto type = global.getElementType();
  auto streamType = llvm::cast<StreamType>(type);
  if (getData().getType() != streamType.getBaseType())
    return emitOpError("data type mismatch");

  AffineMap map = getMap();

  if (map.getNumDims() + map.getNumSymbols() != getIndices().size())
    return emitOpError("affine map dim & symbol count mismatch");

  int64_t rank = global.getShape().size();
  if (map.getNumResults() != rank)
    return emitOpError("affine map result count mismatch: expected ")
           << rank << ", got " << map.getNumResults();

  for (Value idx : getIndices())
    if (!idx.getType().isIndex())
      return emitOpError("indices must be of index type");

  return success();
}

void GlobalStreamPutOp::print(OpAsmPrinter &p) {
  p << " " << getData();
  p << ", @" << getGlobal() << '[';
  if (AffineMapAttr mapAttr =
          (*this)->getAttrOfType<AffineMapAttr>("map")) {
    p.printAffineMapOfSSAIds(mapAttr, getIndices());
  }
  p << ']';
  p.printOptionalAttrDict((*this)->getAttrs(),{"map", "global"});
  p << " : " << getData().getType();
}


ParseResult GlobalStreamPutOp::parse(OpAsmParser &parser,
                                     OperationState &result) {
  auto &builder = parser.getBuilder();
  auto indexTy = builder.getIndexType();
  
  OpAsmParser::UnresolvedOperand data;
  if (parser.parseOperand(data))
    return failure();
  
  if (parser.parseComma()) // ,
    return failure();

  FlatSymbolRefAttr globalAttr;
  if (parser.parseAttribute(globalAttr, "global", result.attributes))
    return failure();

  AffineMapAttr mapAttr;
  SmallVector<OpAsmParser::UnresolvedOperand, 2> mapOperands;
  if (parser.parseAffineMapOfSSAIds(mapOperands, mapAttr, "map", result.attributes))
    return failure();

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  Type dataType;
  if (parser.parseColonType(dataType))
    return failure();

  if (parser.resolveOperands(mapOperands, indexTy, result.operands))
    return failure();
  if (parser.resolveOperand(data, dataType, result.operands))
    return failure();

  return success();
}


void GlobalStreamPutOp::simplifyAffineMap() {
  simplifyStreamAffineMap(*this);
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
// GridMapOp
//===----------------------------------------------------------------------===//
namespace mlir {
namespace allo {

LogicalResult GridMapOp::verify() {
  auto tensors = getTensors();
  auto sharding = getSharding();
  auto grid = getGrid();

  if (tensors.size() != sharding.size())
    return emitOpError() << "number of tensors (" << tensors.size()
                         << ") and sharding lists (" << sharding.size() << ") must match";

  if (!getBody().hasOneBlock())
    return emitOpError() << "region must contain exactly one block";

  Block &bodyBlock = getBody().front();
  if (bodyBlock.getNumArguments() != tensors.size())
    return emitOpError() << "number of block arguments (" << bodyBlock.getNumArguments()
                         << ") and tensors (" << tensors.size() << ") must match";

  for (auto [idx, tensor] : llvm::enumerate(tensors)) {
    auto memrefType = llvm::cast<MemRefType>(tensor.getType());
    auto shardingList = llvm::dyn_cast<ArrayAttr>(sharding[idx]);
    if (!shardingList)
      return emitOpError() << "sharding at index " << idx << " must be an ArrayAttr";
    if (memrefType.getRank() != static_cast<int64_t>(shardingList.size()))
      return emitOpError() << "memref rank (" << memrefType.getRank()
                           << ") and sharding list size (" << shardingList.size()
                           << ") must match for tensor " << idx;
    auto shape = llvm::to_vector<4>(memrefType.getShape());
    for (size_t k = 0; k < shardingList.size(); ++k) {
      auto shardingIntAttr = llvm::dyn_cast<IntegerAttr>(shardingList[k]);
      if (!shardingIntAttr)
        return emitOpError() << "sharding at index " << idx << ", dimension " << k << " must be an IntegerAttr";
      int64_t s = shardingIntAttr.getInt();
      if (s >= static_cast<int64_t>(grid.size()))
        return emitOpError() << "sharding axis " << s << " at index " << idx << " exceeds grid dimension size " << grid.size();
      if (s >= 0) {
        shape[k] = shape[k] / grid[s];
      }
    }
    auto expectedArgType = MemRefType::get(shape, memrefType.getElementType(),
                                           memrefType.getLayout(), memrefType.getMemorySpace());
    if (bodyBlock.getArgument(idx).getType() != expectedArgType)
      return emitOpError() << "block argument " << idx << " type " << bodyBlock.getArgument(idx).getType()
                           << " does not match expected sharded type " << expectedArgType;
  }

  return success();
}

} // namespace allo
} // namespace mlir

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "allo/Dialect/AlloOps.cpp.inc"
