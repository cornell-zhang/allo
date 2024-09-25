/*
 * Copyright HeteroCL authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HCLC_SHARED_LIB_HCL_RUNTIME_UTILS_H
#define HCLC_SHARED_LIB_HCL_RUNTIME_UTILS_H

#ifdef _WIN32
#ifndef HCL_RUNTIME_UTILS_EXPORT
#ifdef hcl_runtime_utils_EXPORTS
// We are building this library
#define HCL_RUNTIME_UTILS_EXPORT __declspec(dllexport)
#define HCL_RUNTIME_UTILS_DEFINE_FUNCTIONS
#else
// We are using this library
#define HCL_RUNTIME_UTILS_EXPORT __declspec(dllimport)
#endif // hcl_runtime_utils_EXPORTS
#endif // HCL_RUNTIME_UTILS_EXPORT
#else  // _WIN32
// Non-windows: use visibility attributes.
#define HCL_RUNTIME_UTILS_EXPORT __attribute__((visibility("default")))
#define HCL_RUNTIME_UTILS_DEFINE_FUNCTIONS
#endif // _WIN32

#include <string>
#include "mlir/ExecutionEngine/CRunnerUtils.h"

extern "C" HCL_RUNTIME_UTILS_EXPORT void
readMemrefI32(int64_t rank, void *ptr, char *str);

extern "C" HCL_RUNTIME_UTILS_EXPORT void
readMemrefI64(int64_t rank, void *ptr, char *str);

extern "C" HCL_RUNTIME_UTILS_EXPORT void
readMemrefF32(int64_t rank, void *ptr, char *str);

extern "C" HCL_RUNTIME_UTILS_EXPORT void
readMemrefF64(int64_t rank, void *ptr, char *str);

extern "C" HCL_RUNTIME_UTILS_EXPORT void
writeMemrefI32(int64_t rank, void *ptr, char *str);

extern "C" HCL_RUNTIME_UTILS_EXPORT void
writeMemrefI64(int64_t rank, void *ptr, char *str);

extern "C" HCL_RUNTIME_UTILS_EXPORT void
writeMemrefF32(int64_t rank, void *ptr, char *str);

extern "C" HCL_RUNTIME_UTILS_EXPORT void
writeMemrefF64(int64_t rank, void *ptr, char *str);

#endif // HCLC_SHARED_LIB_HCL_RUNTIME_UTILS_H