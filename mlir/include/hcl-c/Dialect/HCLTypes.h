/*
 * Copyright HeteroCL authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HCL_MLIR_C_HCLTYPES_H
#define HCL_MLIR_C_HCLTYPES_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_CAPI_EXPORTED bool hclMlirTypeIsALoopHandle(MlirType type);
MLIR_CAPI_EXPORTED MlirType hclMlirLoopHandleTypeGet(MlirContext ctx);

MLIR_CAPI_EXPORTED bool hclMlirTypeIsAOpHandle(MlirType type);
MLIR_CAPI_EXPORTED MlirType hclMlirOpHandleTypeGet(MlirContext ctx);

MLIR_CAPI_EXPORTED bool hclMlirTypeIsAFixedType(MlirType type);
MLIR_CAPI_EXPORTED MlirType hclMlirFixedTypeGet(MlirContext ctx, size_t width, size_t frac);
MLIR_CAPI_EXPORTED unsigned hclMlirFixedTypeGetWidth(MlirType type);
MLIR_CAPI_EXPORTED unsigned hclMlirFixedTypeGetFrac(MlirType type);

MLIR_CAPI_EXPORTED bool hclMlirTypeIsAUFixedType(MlirType type);
MLIR_CAPI_EXPORTED MlirType hclMlirUFixedTypeGet(MlirContext ctx, size_t width, size_t frac);
MLIR_CAPI_EXPORTED unsigned hclMlirUFixedTypeGetWidth(MlirType type);
MLIR_CAPI_EXPORTED unsigned hclMlirUFixedTypeGetFrac(MlirType type);

MLIR_CAPI_EXPORTED bool hclMlirTypeIsAStructType(MlirType type);
MLIR_CAPI_EXPORTED MlirType hclMlirStructTypeGet(MlirContext ctx, intptr_t numElements,
                                            MlirType const *elements);
MLIR_CAPI_EXPORTED MlirType hclMlirStructGetEleType(MlirType type, size_t pos);
MLIR_CAPI_EXPORTED unsigned hclMlirStructTypeGetNumFields(MlirType type);

#ifdef __cplusplus
}
#endif

#endif // HCL_MLIR_C_HCLTYPES_H