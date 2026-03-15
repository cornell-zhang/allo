# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
    MAIN_CONFIG
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(ALLO_TEST_DEPENDS
    FileCheck
    not
    allo-opt
)

add_lit_testsuite(check-allo "Running the allo regression tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${ALLO_TEST_DEPENDS}
)
set_target_properties(check-allo PROPERTIES FOLDER "Tests")

add_lit_testsuites(ALLO ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${ALLO_TEST_DEPENDS})
