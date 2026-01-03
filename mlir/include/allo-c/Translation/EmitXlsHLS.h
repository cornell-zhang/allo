/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_C_TRANSLATION_EMITXlsHLS_H
#define ALLO_C_TRANSLATION_EMITXlsHLS_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_CAPI_EXPORTED MlirLogicalResult mlirEmitXlsHls(MlirModule module,
                                                    MlirStringCallback callback,
                                                    void *userData,
                                                    bool useMemory);

#ifdef __cplusplus
}
#endif

#endif // ALLO_C_TRANSLATION_EMITXlsHLS_H
