// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void add_offset_uint8(std::uint8_t in_buffer[1024], std::uint8_t out_buffer[1024], std::int32_t offset[1]) {
    event0();
    constexpr std::int32_t nbytes = 1024;
    for(int j=0; j<nbytes; j++) {
        out_buffer[j] = in_buffer[j] + offset[0];
    }
    event1();
}

} // extern "C"