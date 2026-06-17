# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 Advanced Micro Devices, Inc.

# parameters
# -DBOOST_ROOT: Path to Boost install
# -DXRT_INC_DIR: Full path to src/runtime_src/core/include in XRT cloned repo
# -DXRT_LIB_DIR: Path to xrt_coreutil.lib
# -DTARGET_NAME: Target name to be built

# cmake needs this line
cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 23) 
set(CMAKE_CXX_STANDARD_REQUIRED YES)

find_program(WSL NAMES powershell.exe)

set(CMAKE_C_COMPILER gcc-13)
set(CMAKE_CXX_COMPILER g++-13)

# -----------------------------------------------------------------------------
# XRT auto-detection (supports both Ubuntu packages and legacy /opt/xilinx/xrt)
#   adapted from the MLIR-AIE project
# -----------------------------------------------------------------------------
if(NOT DEFINED XRT_INC_DIR OR NOT DEFINED XRT_LIB_DIR)
    find_package(XRT QUIET)
    if(XRT_FOUND)
        # find_package(XRT) may resolve via the project's FindXRT.cmake (which
        # sets XRT_INCLUDE_DIR / XRT_LIB_DIR, singular) or via XRT's own
        # xrt-config.cmake (which sets XRT_INCLUDE_DIRS / XRT_LINK_DIRS,
        # plural).  Accept whichever set is available.
        if(NOT DEFINED XRT_INC_DIR)
            if(XRT_INCLUDE_DIRS)
                set(XRT_INC_DIR "${XRT_INCLUDE_DIRS}" CACHE STRING "Path to XRT headers")
            elseif(XRT_INCLUDE_DIR)
                set(XRT_INC_DIR "${XRT_INCLUDE_DIR}" CACHE STRING "Path to XRT headers")
            endif()
        endif()
        if(NOT DEFINED XRT_LIB_DIR)
            if(XRT_LINK_DIRS)
                set(XRT_LIB_DIR "${XRT_LINK_DIRS}" CACHE STRING "Path to XRT libraries")
            endif()
        endif()
    endif()

    # Fall back to legacy/default paths if still unset
    if(NOT DEFINED XRT_INC_DIR OR NOT DEFINED XRT_LIB_DIR)
        find_program(WSL NAMES powershell.exe)
        if(NOT WSL)
            if(NOT DEFINED XRT_INC_DIR)
                set(XRT_INC_DIR /opt/xilinx/xrt/include CACHE STRING "Path to XRT headers")
            endif()
            if(NOT DEFINED XRT_LIB_DIR)
                set(XRT_LIB_DIR /opt/xilinx/xrt/lib CACHE STRING "Path to XRT libraries")
            endif()
        else()
            if(NOT DEFINED XRT_INC_DIR)
                set(XRT_INC_DIR C:/Technical/XRT/src/runtime_src/core/include CACHE STRING "Path to XRT headers")
            endif()
            if(NOT DEFINED XRT_LIB_DIR)
                set(XRT_LIB_DIR C:/Technical/xrtNPUfromDLL CACHE STRING "Path to XRT libraries")
            endif()
        endif()
    endif()
endif()

set(TARGET_NAME test CACHE STRING "Target to be built")
set(MLIR_AIE_DIR test CACHE STRING "MLIR-AIE root directory")

SET (ProjectName ${TARGET_NAME})
SET (currentTarget ${TARGET_NAME})

if ( WSL )
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR})
endif ()

project(${ProjectName})

# Find packages
find_package(Boost REQUIRED)

add_executable(${currentTarget}
    ${MLIR_AIE_DIR}/runtime_lib/test_lib/test_utils.cpp
    test.cpp
)

target_compile_definitions(${currentTarget} PUBLIC DISABLE_ABI_CHECK=1)

target_include_directories (${currentTarget} PUBLIC 
    ${XRT_INC_DIR}
    ${Boost_INCLUDE_DIRS}
    ${MLIR_AIE_DIR}/runtime_lib/test_lib
)

target_link_directories(${currentTarget} PUBLIC
    ${XRT_LIB_DIR}
    ${Boost_LIBRARY_DIRS}
)

if (NOT WSL)
    target_link_libraries(${currentTarget} PUBLIC
        xrt_coreutil
        boost_program_options
        boost_filesystem
    )
else()
    target_link_libraries(${currentTarget} PUBLIC
        xrt_coreutil
    )
endif()
