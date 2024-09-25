# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

add_mlir_library(hcl_runtime_utils
    SHARED
    HCLRuntimeUtils.cpp
)
target_compile_definitions(hcl_runtime_utils PRIVATE hcl_runtime_utils_EXPORTS)