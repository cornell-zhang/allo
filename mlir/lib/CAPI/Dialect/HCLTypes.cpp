/*
 * Copyright HeteroCL authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hcl-c/Dialect/HCLTypes.h"
#include "hcl/Dialect/HeteroCLTypes.h"
#include "mlir/CAPI/Registration.h"

using namespace mlir;
using namespace hcl;

bool hclMlirTypeIsALoopHandle(MlirType type) {
  return unwrap(type).isa<hcl::LoopHandleType>();
}

MlirType hclMlirLoopHandleTypeGet(MlirContext ctx) {
  return wrap(hcl::LoopHandleType::get(unwrap(ctx)));
}

bool hclMlirTypeIsAOpHandle(MlirType type) {
  return unwrap(type).isa<hcl::OpHandleType>();
}

MlirType hclMlirOpHandleTypeGet(MlirContext ctx) {
  return wrap(hcl::OpHandleType::get(unwrap(ctx)));
}

bool hclMlirTypeIsAFixedType(MlirType type) {
  return unwrap(type).isa<hcl::FixedType>();
}

MlirType hclMlirFixedTypeGet(MlirContext ctx, size_t width, size_t frac) {
  return wrap(hcl::FixedType::get(unwrap(ctx), width, frac));
}

unsigned hclMlirFixedTypeGetWidth(MlirType type) {
  return unwrap(type).cast<hcl::FixedType>().getWidth();
}

unsigned hclMlirFixedTypeGetFrac(MlirType type) {
  return unwrap(type).cast<hcl::FixedType>().getFrac();
}

bool hclMlirTypeIsAUFixedType(MlirType type) {
  return unwrap(type).isa<hcl::UFixedType>();
}

MlirType hclMlirUFixedTypeGet(MlirContext ctx, size_t width, size_t frac) {
  return wrap(hcl::UFixedType::get(unwrap(ctx), width, frac));
}

unsigned hclMlirUFixedTypeGetWidth(MlirType type) {
  return unwrap(type).cast<hcl::UFixedType>().getWidth();
}

unsigned hclMlirUFixedTypeGetFrac(MlirType type) {
  return unwrap(type).cast<hcl::UFixedType>().getFrac();
}

bool hclMlirTypeIsAStructType(MlirType type) {
  return unwrap(type).isa<hcl::StructType>();
}

MlirType hclMlirStructTypeGet(MlirContext ctx, intptr_t numElements,
                              MlirType const *elements) {
  SmallVector<Type, 4> types;
  ArrayRef<Type> typeRef = unwrapList(numElements, elements, types);
  return wrap(hcl::StructType::get(unwrap(ctx), typeRef));
}

MlirType hclMlirStructGetEleType(MlirType type, size_t pos) {
  return wrap(unwrap(type).cast<hcl::StructType>().getElementTypes()[pos]);
}

unsigned hclMlirStructTypeGetNumFields(MlirType type) {
  return unwrap(type).cast<hcl::StructType>().getElementTypes().size();
}