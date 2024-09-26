/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLOC_SHARED_LIB_ALLO_RUNTIME_UTILS_H
#define ALLOC_SHARED_LIB_ALLO_RUNTIME_UTILS_H

#ifdef _WIN32
#ifndef ALLO_RUNTIME_UTILS_EXPORT
#ifdef allo_runtime_utils_EXPORTS
// We are building this library
#define ALLO_RUNTIME_UTILS_EXPORT __declspec(dllexport)
#define ALLO_RUNTIME_UTILS_DEFINE_FUNCTIONS
#else
// We are using this library
#define ALLO_RUNTIME_UTILS_EXPORT __declspec(dllimport)
#endif // allo_runtime_utils_EXPORTS
#endif // ALLO_RUNTIME_UTILS_EXPORT
#else  // _WIN32
// Non-windows: use visibility attributes.
#define ALLO_RUNTIME_UTILS_EXPORT __attribute__((visibility("default")))
#define ALLO_RUNTIME_UTILS_DEFINE_FUNCTIONS
#endif // _WIN32

#include <string>
#include "mlir/ExecutionEngine/CRunnerUtils.h"

extern "C" ALLO_RUNTIME_UTILS_EXPORT void
readMemrefI32(int64_t rank, void *ptr, char *str);

extern "C" ALLO_RUNTIME_UTILS_EXPORT void
readMemrefI64(int64_t rank, void *ptr, char *str);

extern "C" ALLO_RUNTIME_UTILS_EXPORT void
readMemrefF32(int64_t rank, void *ptr, char *str);

extern "C" ALLO_RUNTIME_UTILS_EXPORT void
readMemrefF64(int64_t rank, void *ptr, char *str);

extern "C" ALLO_RUNTIME_UTILS_EXPORT void
writeMemrefI32(int64_t rank, void *ptr, char *str);

extern "C" ALLO_RUNTIME_UTILS_EXPORT void
writeMemrefI64(int64_t rank, void *ptr, char *str);

extern "C" ALLO_RUNTIME_UTILS_EXPORT void
writeMemrefF32(int64_t rank, void *ptr, char *str);

extern "C" ALLO_RUNTIME_UTILS_EXPORT void
writeMemrefF64(int64_t rank, void *ptr, char *str);

#endif // ALLOC_SHARED_LIB_ALLO_RUNTIME_UTILS_H