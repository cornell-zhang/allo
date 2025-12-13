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

      if (auto boolAttr = llvm::dyn_cast<BoolAttr>(constAttr)) {
        return SmallString<8>(std::to_string(boolAttr.getValue()));

      } else if (auto floatAttr = llvm::dyn_cast<FloatAttr>(constAttr)) {
        // as 0.0 will be interpreted as double constant, we need to explicitly
        // declare it as float32
        int bitwidth =
            llvm::dyn_cast<FloatType>(floatAttr.getType()).getWidth();
        std::string prefix = (bitwidth == 32) ? "(float)" : "(double)";
        auto value = floatAttr.getValueAsDouble();
        if (std::isfinite(value))
          return SmallString<8>(prefix + std::to_string(value));
        else if (value > 0)
          return SmallString<8>("INFINITY");
        else
          return SmallString<8>("-INFINITY");

      } else if (auto intAttr = llvm::dyn_cast<IntegerAttr>(constAttr)) {
        auto value = intAttr.getInt();
        return SmallString<8>(std::to_string(value));
      }
    }
  }
  return state.nameTable.lookup(val);
};

Type getUnsignedTypeFromSigned(Type type) {
  if (auto intType = llvm::dyn_cast<IntegerType>(type)) {
    return IntegerType::get(type.getContext(), intType.getWidth(),
                            IntegerType::SignednessSemantics::Unsigned);
  } else if (auto memrefType = llvm::dyn_cast<MemRefType>(type)) {
    Type elt = getUnsignedTypeFromSigned(memrefType.getElementType());
    return MemRefType::get(memrefType.getShape(), elt, memrefType.getLayout(),
                           memrefType.getMemorySpace());
  } else if (auto streamType = llvm::dyn_cast<StreamType>(type)) {
    Type elt = getUnsignedTypeFromSigned(streamType.getBaseType());
    return StreamType::get(type.getContext(), elt, streamType.getDepth());
  }
  return type;
}

void fixUnsignedType(Value &result, bool isUnsigned) {
  if (isUnsigned) {
    result.setType(getUnsignedTypeFromSigned(result.getType()));
  }
}

void fixUnsignedType(memref::GlobalOp &op, bool isUnsigned) {
  if (isUnsigned) { // unsigned type
    auto type = op.getTypeAttr().getValue();
    op.setTypeAttr(TypeAttr::get(getUnsignedTypeFromSigned(type)));
  }
}