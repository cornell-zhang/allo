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
  return unwrap(type).isa<allo::LoopHandleType>();
}

MlirType alloMlirLoopHandleTypeGet(MlirContext ctx) {
  return wrap(allo::LoopHandleType::get(unwrap(ctx)));
}

bool alloMlirTypeIsAOpHandle(MlirType type) {
  return unwrap(type).isa<allo::OpHandleType>();
}

MlirType alloMlirOpHandleTypeGet(MlirContext ctx) {
  return wrap(allo::OpHandleType::get(unwrap(ctx)));
}

bool alloMlirTypeIsAFixedType(MlirType type) {
  return unwrap(type).isa<allo::FixedType>();
}

MlirType alloMlirFixedTypeGet(MlirContext ctx, size_t width, size_t frac) {
  return wrap(allo::FixedType::get(unwrap(ctx), width, frac));
}

unsigned alloMlirFixedTypeGetWidth(MlirType type) {
  return unwrap(type).cast<allo::FixedType>().getWidth();
}

unsigned alloMlirFixedTypeGetFrac(MlirType type) {
  return unwrap(type).cast<allo::FixedType>().getFrac();
}

bool alloMlirTypeIsAUFixedType(MlirType type) {
  return unwrap(type).isa<allo::UFixedType>();
}

MlirType alloMlirUFixedTypeGet(MlirContext ctx, size_t width, size_t frac) {
  return wrap(allo::UFixedType::get(unwrap(ctx), width, frac));
}

unsigned alloMlirUFixedTypeGetWidth(MlirType type) {
  return unwrap(type).cast<allo::UFixedType>().getWidth();
}

unsigned alloMlirUFixedTypeGetFrac(MlirType type) {
  return unwrap(type).cast<allo::UFixedType>().getFrac();
}

bool alloMlirTypeIsAStructType(MlirType type) {
  return unwrap(type).isa<allo::StructType>();
}

MlirType alloMlirStructTypeGet(MlirContext ctx, intptr_t numElements,
                               MlirType const *elements) {
  SmallVector<Type, 4> types;
  ArrayRef<Type> typeRef = unwrapList(numElements, elements, types);
  return wrap(allo::StructType::get(unwrap(ctx), typeRef));
}

MlirType alloMlirStructGetEleType(MlirType type, size_t pos) {
  return wrap(unwrap(type).cast<allo::StructType>().getElementTypes()[pos]);
}

unsigned alloMlirStructTypeGetNumFields(MlirType type) {
  return unwrap(type).cast<allo::StructType>().getElementTypes().size();
}

bool alloMlirTypeIsAStreamType(MlirType type) {
  return unwrap(type).isa<allo::StreamType>();
}

MlirType alloMlirStreamTypeGet(MlirContext ctx, MlirType baseType,
                               size_t depth) {
  return wrap(allo::StreamType::get(unwrap(ctx), unwrap(baseType), depth));
}

MlirType alloMlirStreamTypeGetBaseType(MlirType type) {
  return wrap(unwrap(type).cast<allo::StreamType>().getBaseType());
}

unsigned alloMlirStreamTypeGetDepth(MlirType type) {
  return unwrap(type).cast<allo::StreamType>().getDepth();
}
