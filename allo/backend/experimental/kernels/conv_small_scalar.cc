/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void conv2d_int32(int32_t input[3][3], int32_t kernel[3][3],
                  int32_t output[5][5]) {
  // Direct implementation - flatten arrays for easier access
  int32_t *in = reinterpret_cast<int32_t *>(input);
  int32_t *ker = reinterpret_cast<int32_t *>(kernel);
  int32_t *out = reinterpret_cast<int32_t *>(output);

  // Constants for readability
  constexpr int IN_WIDTH = 3;
  constexpr int IN_HEIGHT = 3;
  constexpr int OUT_WIDTH = 5;
  constexpr int OUT_HEIGHT = 5;
  constexpr int KERNEL_SIZE = 3;
  constexpr int PAD = 2;
  // Add event markers for profiling
  event0();
  // 2D Convolution with padding: output[i][j] = sum over kernel of
  // input[i+ki-pad][j+kj-pad] * kernel[ki][kj]
  for (int out_row = 0; out_row < OUT_HEIGHT; out_row++) {
    for (int out_col = 0; out_col < OUT_WIDTH; out_col++) {
      int32_t sum = 0; // Use int32 to avoid overflow during accumulation

      // Convolve with 3x3 kernel
      for (int k_row = 0; k_row < KERNEL_SIZE; k_row++) {
        for (int k_col = 0; k_col < KERNEL_SIZE; k_col++) {
          // Calculate input coordinates with padding
          int in_row = out_row + k_row - PAD;
          int in_col = out_col + k_col - PAD;

          // Check bounds - use 0 for padding
          int32_t input_val = 0;
          if (in_row >= 0 && in_row < IN_HEIGHT && in_col >= 0 &&
              in_col < IN_WIDTH) {
            input_val = in[in_row * IN_WIDTH + in_col];
          }

          // Perform multiply-accumulate
          int32_t kernel_val = ker[k_row * KERNEL_SIZE + k_col];
          sum += input_val * kernel_val;
        }
      }

      // Store result
      out[out_row * OUT_WIDTH + out_col] = sum;
    }
  }
  event1();
}

} // extern "C"