# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

add_mlir_library(allo_runtime_utils
    SHARED
    AlloRuntimeUtils.cpp
)
target_compile_definitions(allo_runtime_utils PRIVATE allo_runtime_utils_EXPORTS)