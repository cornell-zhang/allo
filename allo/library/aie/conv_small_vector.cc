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
  // Input dimensions
  constexpr int IN_H = 3;
  constexpr int IN_W = 3;
  // Kernel dimensions
  constexpr int K_H = 3;
  constexpr int K_W = 3;
  // Padding
  constexpr int PADDING = 2;
  // Padded input dimensions
  constexpr int PADDED_IN_H = IN_H + 2 * PADDING; // 3 + 2*2 = 7
  constexpr int PADDED_IN_W = IN_W + 2 * PADDING; // 3 + 2*2 = 7
  // Output dimensions
  constexpr int OUT_H = IN_H - K_H + 2 * PADDING + 1; // 3 - 3 + 4 + 1 = 5
  constexpr int OUT_W = IN_W - K_W + 2 * PADDING + 1; // 3 - 3 + 4 + 1 = 5

  // Vectorization factor for int32_t. AIE-ML/XDNA 1 supports 4, 8, 16, 32.
  // VEC_SIZE = 4 is chosen as it aligns well with OUT_W=5 (4 elements + 1 tail
  // element) and the native 128-bit vector operations for int32 (4 * 4 bytes =
  // 16 bytes).
  constexpr int VEC_SIZE = 4;

  // Padded input buffer (7x7)
  // Ensure alignment for AIE vector operations.
  alignas(aie::vector_decl_align)
      int32_t padded_input[PADDED_IN_H][PADDED_IN_W] = {0};

  // Copy original input to the center of padded_input
  for (int i = 0; i < IN_H; ++i) {
    for (int j = 0; j < IN_W; ++j) {
      padded_input[i + PADDING][j + PADDING] = input[i][j];
    }
  }

  // Pre-splat kernel values into vectors
  // This avoids creating a new splatted vector in the inner loop for each MAC
  // operation. This array will hold 9 vectors, each containing the same kernel
  // value replicated VEC_SIZE times.
  aie::vector<int32_t, VEC_SIZE> k_vec[K_H][K_W];
  for (int r = 0; r < K_H; ++r) {
    for (int c = 0; c < K_W; ++c) {
      // Reverting to loop-based splatting as aie::broadcast is not directly
      // available or has a different signature in the provided API.
      for (int i = 0; i < VEC_SIZE; ++i) {
        k_vec[r][c][i] = kernel[r][c];
      }
    }
  }

  // Add event markers for profiling
  event0();

  // Loop over output rows
  for (int out_row = 0; out_row < OUT_H; ++out_row) {
    // Loop over output columns, processing VEC_SIZE elements at a time
    // The last iteration will handle the tail (OUT_W % VEC_SIZE) elements.
    for (int out_col = 0; out_col < OUT_W; out_col += VEC_SIZE) {
      // Declare accumulator for VEC_SIZE output pixels.
      // acc64 is the default accumulator type for int32*int32 multiplication on
      // AIE-ML/XDNA 1.
      aie::accum<acc64, VEC_SIZE> acc_vec;

      // Manually unroll the K_H (kernel height) and K_W (kernel width) loops.
      // This eliminates loop overhead and allows the compiler to schedule
      // the 9 multiply-accumulate operations more aggressively.
      // This is highly effective for small, fixed kernel sizes.

      // K_H = 0
      {
        // K_W = 0
        aie::vector<int32_t, VEC_SIZE> v_input_window_00 =
            aie::load_unaligned_v<VEC_SIZE>(
                &padded_input[out_row + 0][out_col + 0], 1);
        acc_vec = aie::mul(v_input_window_00, k_vec[0][0]);

        // K_W = 1
        aie::vector<int32_t, VEC_SIZE> v_input_window_01 =
            aie::load_unaligned_v<VEC_SIZE>(
                &padded_input[out_row + 0][out_col + 1], 1);
        acc_vec = aie::mac(acc_vec, v_input_window_01, k_vec[0][1]);

        // K_W = 2
        aie::vector<int32_t, VEC_SIZE> v_input_window_02 =
            aie::load_unaligned_v<VEC_SIZE>(
                &padded_input[out_row + 0][out_col + 2], 1);
        acc_vec = aie::mac(acc_vec, v_input_window_02, k_vec[0][2]);
      }
      // K_H = 1
      {
        // K_W = 0
        aie::vector<int32_t, VEC_SIZE> v_input_window_10 =
            aie::load_unaligned_v<VEC_SIZE>(
                &padded_input[out_row + 1][out_col + 0], 1);
        acc_vec = aie::mac(acc_vec, v_input_window_10, k_vec[1][0]);

        // K_W = 1
        aie::vector<int32_t, VEC_SIZE> v_input_window_11 =
            aie::load_unaligned_v<VEC_SIZE>(
                &padded_input[out_row + 1][out_col + 1], 1);
        acc_vec = aie::mac(acc_vec, v_input_window_11, k_vec[1][1]);

        // K_W = 2
        aie::vector<int32_t, VEC_SIZE> v_input_window_12 =
            aie::load_unaligned_v<VEC_SIZE>(
                &padded_input[out_row + 1][out_col + 2], 1);
        acc_vec = aie::mac(acc_vec, v_input_window_12, k_vec[1][2]);
      }
      // K_H = 2
      {
        // K_W = 0
        aie::vector<int32_t, VEC_SIZE> v_input_window_20 =
            aie::load_unaligned_v<VEC_SIZE>(
                &padded_input[out_row + 2][out_col + 0], 1);
        acc_vec = aie::mac(acc_vec, v_input_window_20, k_vec[2][0]);

        // K_W = 1
        aie::vector<int32_t, VEC_SIZE> v_input_window_21 =
            aie::load_unaligned_v<VEC_SIZE>(
                &padded_input[out_row + 2][out_col + 1], 1);
        acc_vec = aie::mac(acc_vec, v_input_window_21, k_vec[2][1]);

        // K_W = 2
        aie::vector<int32_t, VEC_SIZE> v_input_window_22 =
            aie::load_unaligned_v<VEC_SIZE>(
                &padded_input[out_row + 2][out_col + 2], 1);
        acc_vec = aie::mac(acc_vec, v_input_window_22, k_vec[2][2]);
      }

      // Store the final accumulated results to the output tensor.
      // Handle potential partial vector at the end of a row.
      if (out_col + VEC_SIZE <= OUT_W) {
        // If a full vector can be stored, use aie::store_unaligned_v.
        // The '1' indicates that the pointer is 1-element (4-byte) aligned.
        aie::store_unaligned_v(&output[out_row][out_col],
                               acc_vec.to_vector<int32_t>(), 1);
      } else {
        // This handles the "tail" case where fewer than VEC_SIZE elements
        // remain. For OUT_W=5 and VEC_SIZE=4, this will handle the last element
        // (index 4).
        aie::vector<int32_t, VEC_SIZE> temp_out_vec =
            acc_vec.to_vector<int32_t>();
        for (int i = 0; i < (OUT_W - out_col); ++i) {
          // Store element by element for the remainder.
          output[out_row][out_col + i] = temp_out_vec[i];
        }
      }
    }
  }
  event1();
}

} // extern "C"