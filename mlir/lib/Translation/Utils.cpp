/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Translation/Utils.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace allo;

// TODO: update naming rule.
SmallString<8> AlloEmitterBase::addName(Value val, bool isPtr,
                                        std::string name) {
  assert(!isDeclared(val) && "has been declared before.");

  SmallString<8> valName;
  if (isPtr)
    valName += "*";

  if (name != "") {
    if (state.nameConflictCnt.count(name) > 0) {
      state.nameConflictCnt[name]++;
      valName += StringRef(name + std::to_string(state.nameConflictCnt[name]));
    } else { // first time
      state.nameConflictCnt[name] = 0;
      valName += name;
    }
  } else {
    valName += StringRef("v" + std::to_string(state.nameTable.size()));
  }
  state.nameTable[val] = valName;

  return valName;
};

SmallString<8> AlloEmitterBase::getName(Value val) {
  // For constant scalar operations, the constant number will be returned
  // rather than the value name.
  if (auto defOp = val.getDefiningOp()) {
    if (auto constOp = dyn_cast<arith::ConstantOp>(defOp)) {
      auto constAttr = constOp.getValue();

      if (auto boolAttr = constAttr.dyn_cast<BoolAttr>()) {
        return SmallString<8>(std::to_string(boolAttr.getValue()));

      } else if (auto floatAttr = constAttr.dyn_cast<FloatAttr>()) {
        // as 0.0 will be interpreted as double constant, we need to explicitly
        // declare it as float32
        int bitwidth = floatAttr.getType().cast<FloatType>().getWidth();
        std::string prefix = (bitwidth == 32) ? "(float)" : "(double)";
        auto value = floatAttr.getValueAsDouble();
        if (std::isfinite(value))
          return SmallString<8>(prefix + std::to_string(value));
        else if (value > 0)
          return SmallString<8>("INFINITY");
        else
          return SmallString<8>("-INFINITY");

      } else if (auto intAttr = constAttr.dyn_cast<IntegerAttr>()) {
        auto value = intAttr.getInt();
        return SmallString<8>(std::to_string(value));
      }
    }
  }
  return state.nameTable.lookup(val);
};

void fixUnsignedType(Value &result, bool isUnsigned) {
  if (isUnsigned) { // unsigned type
    if (result.getType().isa<MemRefType>()) {
      auto arrayType = result.getType().dyn_cast<MemRefType>();
      Type elt = IntegerType::get(
          arrayType.getContext(),
          arrayType.getElementType().cast<IntegerType>().getWidth(),
          IntegerType::SignednessSemantics::Unsigned);
      result.setType(MemRefType::get(arrayType.getShape(), elt,
                                     arrayType.getLayout(),
                                     arrayType.getMemorySpace()));
    } else if (result.getType().isa<IntegerType>()) {
      Type type =
          IntegerType::get(result.getType().getContext(),
                           result.getType().cast<IntegerType>().getWidth(),
                           IntegerType::SignednessSemantics::Unsigned);
      result.setType(type);
    }
  }
}

void fixUnsignedType(memref::GlobalOp &op, bool isUnsigned) {
  if (isUnsigned) { // unsigned type
    auto type = op.getTypeAttr().getValue();
    if (type.isa<MemRefType>()) {
      auto arrayType = type.dyn_cast<MemRefType>();
      Type elt = IntegerType::get(
          arrayType.getContext(),
          arrayType.getElementType().cast<IntegerType>().getWidth(),
          IntegerType::SignednessSemantics::Unsigned);
      // get a memref type attr
      op.setTypeAttr(TypeAttr::get(
          MemRefType::get(arrayType.getShape(), elt, arrayType.getLayout(),
                          arrayType.getMemorySpace())));
    } else if (type.isa<IntegerType>()) {
      Type type = IntegerType::get(type.getContext(),
                                   type.cast<IntegerType>().getWidth(),
                                   IntegerType::SignednessSemantics::Unsigned);
      op.setTypeAttr(TypeAttr::get(type));
    }
  }
}