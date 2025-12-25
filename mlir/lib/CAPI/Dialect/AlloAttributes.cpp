/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo-c/Dialect/AlloAttributes.h"
#include "allo/Dialect/AlloAttrs.h"
#include "allo/Dialect/AlloDialect.h"
#include "mlir/CAPI/Registration.h"

using namespace mlir;
using namespace allo;

bool mlirAttributeIsAPartitionKind(MlirAttribute attr) {
  return llvm::isa<PartitionKindEnumAttr>(unwrap(attr));
}

MlirAttribute mlirPartitionKindGet(MlirContext ctx, MlirAttribute kind) {
  IntegerAttr attr = llvm::dyn_cast<IntegerAttr>(unwrap(kind));
  PartitionKindEnum kindEnum = static_cast<PartitionKindEnum>(attr.getInt());
  return wrap(PartitionKindEnumAttr::get(unwrap(ctx), kindEnum));
}

bool mlirAttributeIsANDRangeDimKind(MlirAttribute attr) {
  return llvm::isa<NDRangeDimKindEnumAttr>(unwrap(attr));
}

MlirAttribute mlirNDRangeDimKindGet(MlirContext ctx, MlirAttribute kind) {
  IntegerAttr attr = llvm::dyn_cast<IntegerAttr>(unwrap(kind));
  NDRangeDimKindEnum kindEnum = static_cast<NDRangeDimKindEnum>(attr.getInt());
  return wrap(NDRangeDimKindEnumAttr::get(unwrap(ctx), kindEnum));
}