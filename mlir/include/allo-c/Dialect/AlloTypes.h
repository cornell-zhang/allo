/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_MLIR_C_TYPES__H
#define ALLO_MLIR_C_TYPES__H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_CAPI_EXPORTED bool alloMlirTypeIsALoopHandle(MlirType type);
MLIR_CAPI_EXPORTED MlirType alloMlirLoopHandleTypeGet(MlirContext ctx);

MLIR_CAPI_EXPORTED bool alloMlirTypeIsAOpHandle(MlirType type);
MLIR_CAPI_EXPORTED MlirType alloMlirOpHandleTypeGet(MlirContext ctx);

MLIR_CAPI_EXPORTED bool alloMlirTypeIsAFixedType(MlirType type);
MLIR_CAPI_EXPORTED MlirType alloMlirFixedTypeGet(MlirContext ctx, size_t width,
                                                 size_t frac);
MLIR_CAPI_EXPORTED unsigned alloMlirFixedTypeGetWidth(MlirType type);
MLIR_CAPI_EXPORTED unsigned alloMlirFixedTypeGetFrac(MlirType type);

MLIR_CAPI_EXPORTED bool alloMlirTypeIsAUFixedType(MlirType type);
MLIR_CAPI_EXPORTED MlirType alloMlirUFixedTypeGet(MlirContext ctx, size_t width,
                                                  size_t frac);
MLIR_CAPI_EXPORTED unsigned alloMlirUFixedTypeGetWidth(MlirType type);
MLIR_CAPI_EXPORTED unsigned alloMlirUFixedTypeGetFrac(MlirType type);

MLIR_CAPI_EXPORTED bool alloMlirTypeIsAStructType(MlirType type);
MLIR_CAPI_EXPORTED MlirType alloMlirStructTypeGet(MlirContext ctx,
                                                  intptr_t numElements,
                                                  MlirType const *elements);
MLIR_CAPI_EXPORTED MlirType alloMlirStructGetEleType(MlirType type, size_t pos);
MLIR_CAPI_EXPORTED unsigned alloMlirStructTypeGetNumFields(MlirType type);

MLIR_CAPI_EXPORTED bool alloMlirTypeIsAStreamType(MlirType type);
MLIR_CAPI_EXPORTED MlirType alloMlirStreamTypeGet(MlirContext ctx,
                                                  MlirType baseType,
                                                  size_t depth);
MLIR_CAPI_EXPORTED MlirType alloMlirStreamTypeGetBaseType(MlirType type);
MLIR_CAPI_EXPORTED unsigned alloMlirStreamTypeGetDepth(MlirType type);

#ifdef __cplusplus
}
#endif

#endif // ALLO_MLIR_C_TYPES__H