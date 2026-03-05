# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

nanobind_add_module(_liballo
    NB_STATIC STABLE_ABI
    ir.cpp
    ops.cpp
    main.cpp
    transform.cpp
    allo.cpp
)

# turn off -Wcast-qual for nanobind
target_compile_options(nanobind-static
    PUBLIC
    "-Wno-cast-qual"
    "-Wno-covered-switch-default"
)

set_target_properties(_liballo PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/../allo/bindings"
    CXX_VISIBILITY_PRESET hidden
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

target_link_libraries(_liballo
    PRIVATE
    # MLIR Libraries
    ${dialect_libs}
    ${conversion_libs}

    MLIRAllo
    MLIRAlloTransformOps

    Python3::Module
)

target_link_options(_liballo PRIVATE ${LLVM_LDFLAGS})
