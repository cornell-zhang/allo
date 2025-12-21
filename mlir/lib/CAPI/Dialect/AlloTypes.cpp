/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo-c/Dialect/AlloTypes.h"
#include "allo/Dialect/AlloTypes.h"
#include "mlir/CAPI/Registration.h"

using namespace mlir;
using namespace allo;

bool alloMlirTypeIsALoopHandle(MlirType type) {
  return llvm::isa<allo::LoopHandleType>(unwrap(type));
}

MlirType alloMlirLoopHandleTypeGet(MlirContext ctx) {
  return wrap(allo::LoopHandleType::get(unwrap(ctx)));
}

bool alloMlirTypeIsAOpHandle(MlirType type) {
  return llvm::isa<allo::OpHandleType>(unwrap(type));
}

MlirType alloMlirOpHandleTypeGet(MlirContext ctx) {
  return wrap(allo::OpHandleType::get(unwrap(ctx)));
}

bool alloMlirTypeIsAFixedType(MlirType type) {
  return llvm::isa<allo::FixedType>(unwrap(type));
}

MlirType alloMlirFixedTypeGet(MlirContext ctx, size_t width, size_t frac) {
  return wrap(allo::FixedType::get(unwrap(ctx), width, frac));
}

unsigned alloMlirFixedTypeGetWidth(MlirType type) {
  return llvm::dyn_cast<allo::FixedType>(unwrap(type)).getWidth();
}

unsigned alloMlirFixedTypeGetFrac(MlirType type) {
  return llvm::dyn_cast<allo::FixedType>(unwrap(type)).getFrac();
}

bool alloMlirTypeIsAUFixedType(MlirType type) {
  return llvm::isa<allo::UFixedType>(unwrap(type));
}

MlirType alloMlirUFixedTypeGet(MlirContext ctx, size_t width, size_t frac) {
  return wrap(allo::UFixedType::get(unwrap(ctx), width, frac));
}

unsigned alloMlirUFixedTypeGetWidth(MlirType type) {
  return llvm::dyn_cast<allo::UFixedType>(unwrap(type)).getWidth();
}

unsigned alloMlirUFixedTypeGetFrac(MlirType type) {
  return llvm::dyn_cast<allo::UFixedType>(unwrap(type)).getFrac();
}

bool alloMlirTypeIsAStructType(MlirType type) {
  return llvm::isa<allo::StructType>(unwrap(type));
}

MlirType alloMlirStructTypeGet(MlirContext ctx, intptr_t numElements,
                               MlirType const *elements) {
  SmallVector<Type, 4> types;
  ArrayRef<Type> typeRef = unwrapList(numElements, elements, types);
  return wrap(allo::StructType::get(unwrap(ctx), typeRef));
}

MlirType alloMlirStructGetEleType(MlirType type, size_t pos) {
  return wrap(
      llvm::dyn_cast<allo::StructType>(unwrap(type)).getElementTypes()[pos]);
}

unsigned alloMlirStructTypeGetNumFields(MlirType type) {
  return llvm::dyn_cast<allo::StructType>(unwrap(type))
      .getElementTypes()
      .size();
}

bool alloMlirTypeIsAStreamType(MlirType type) {
  return llvm::isa<allo::StreamType>(unwrap(type));
}

MlirType alloMlirStreamTypeGet(MlirContext ctx, MlirType baseType,
                               size_t depth) {
  return wrap(allo::StreamType::get(unwrap(ctx), unwrap(baseType), depth));
}

MlirType alloMlirStreamTypeGetBaseType(MlirType type) {
  return wrap(llvm::dyn_cast<allo::StreamType>(unwrap(type)).getBaseType());
}

unsigned alloMlirStreamTypeGetDepth(MlirType type) {
  return llvm::dyn_cast<allo::StreamType>(unwrap(type)).getDepth();
}
