/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <aie_api/aie.hpp>
#include <aie_api/operators.hpp> // For operator overloading
#include <algorithm>             // For std::max
#include <limits>                // For std::numeric_limits
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>
#define NOCPP

extern "C" {

void masked_softmax_float32(float attention_score[32][64],
                            int tile_row_start[1], float attn_weights[32][64]) {
  // Use aie::operators namespace for convenient syntax
  using namespace aie::operators;
  // Define constants for tile dimensions and vectorization
  constexpr int TILE_ROWS = 32;
  constexpr int SEQ_COLS = 64;
  constexpr int VEC_SIZE = 32;

  // Constants for exp(x) = 2^(x * log2(e))
  // log2(e) = 1 / ln(2) = 1.4426950408889634
  // ln(2) = 0.6931471805599453
  constexpr float LOG2E_F = 1.4426950408889634f;
  constexpr float LN2_F = 0.6931471805599453f;

  float data[128];
  for (int i = 0; i < 128; ++i) {
    float val = 1.0f;
    int exponent = i - 127; // Exponent from -127 to 0
    if (exponent < 0) {
      for (int j = 0; j < -exponent; ++j) {
        val /= 2.0f;
      }
    } else { // exponent >= 0
      for (int j = 0; j < exponent; ++j) {
        val *= 2.0f;
      }
    }
    data[i] = val;
  }
  // Pre-calculate column index vectors for masking.
  // These are constant across all rows and can be initialized once.
  aie::vector<int, VEC_SIZE> col_indices_0;
  for (int k = 0; k < VEC_SIZE; ++k) {
    col_indices_0[k] = k;
  }
  aie::vector<int, VEC_SIZE> col_indices_1;
  for (int k = 0; k < VEC_SIZE; ++k) {
    col_indices_1[k] = k + VEC_SIZE;
  }

  // Define negative infinity constant
  const float neg_inf = -std::numeric_limits<float>::infinity();
  // Create a 32-element vector filled with negative infinity.
  aie::vector<float, VEC_SIZE> neg_inf_vec =
      aie::broadcast<float, VEC_SIZE>(neg_inf);

  // Loop over each row in the tile
  for (int r = 0; r < TILE_ROWS; ++r) {
    // Calculate global row index for causal masking
    int global_row_idx = tile_row_start[0] + r;

    // Create a 32-element vector filled with the global_row_idx for
    // comparison.
    aie::vector<int, VEC_SIZE> global_row_idx_vec =
        aie::broadcast<int, VEC_SIZE>(global_row_idx);

    // Pointers for current row's input and output
    float *__restrict current_attention_score_row_ptr = &attention_score[r][0];
    float *__restrict current_attn_weights_row_ptr = &attn_weights[r][0];

    // Load the two vector segments for the current row (64 columns / 32
    // elements per vector = 2 vectors)
    aie::vector<float, VEC_SIZE> scores_v0 =
        aie::load_v<VEC_SIZE>(current_attention_score_row_ptr);
    aie::vector<float, VEC_SIZE> scores_v1 =
        aie::load_v<VEC_SIZE>(current_attention_score_row_ptr + VEC_SIZE);

    // --- Apply Causal Masking ---
    // Mask for the first vector segment (columns 0 to 31)
    // If column index > global_row_idx, set the score to -infinity
    aie::mask<VEC_SIZE> mask_v0 = col_indices_0 > global_row_idx_vec;
    scores_v0 = aie::select(scores_v0, neg_inf_vec, mask_v0);
    aie::mask<VEC_SIZE> mask_v1 = col_indices_1 > global_row_idx_vec;
    scores_v1 = aie::select(scores_v1, neg_inf_vec, mask_v1);

    // --- Find Max Value for Numerical Stability (LogSumExp trick) ---
    // aie::reduce_max should work for float vectors.
    float row_max = aie::reduce_max(scores_v0);
    row_max = std::max(row_max, aie::reduce_max(scores_v1));

    // Create a 32-element vector filled with the row_max for element-wise
    // subtraction.
    aie::vector<float, VEC_SIZE> row_max_vec =
        aie::broadcast<float, VEC_SIZE>(-row_max);
    scores_v0 = aie::add(scores_v0, row_max_vec);
    scores_v1 = aie::add(scores_v1, row_max_vec);
    // --- Compute exp(x - max) using scalar approximation ---
    aie::vector<float, VEC_SIZE> exp_scores_v0;
    for (int k = 0; k < VEC_SIZE; ++k) {
      if (scores_v0[k] < -80.0f) {
        exp_scores_v0[k] = 0.0f;
      } else {
        float y = scores_v0[k] * LOG2E_F;
        int I = static_cast<int>(y);
        if (y < 0.0f && y != static_cast<float>(I)) {
          I--;
        }
        float F = y - static_cast<float>(I);
        float pow2_I;
        if (I < -127) {
          pow2_I = 0.0f; // Below table range, effectively zero
        } else if (I > 0) {
          pow2_I = data[127]; // Index 127 corresponds to I=0
        } else {              // I is in [-127, 0]
          pow2_I = data[I + 127];
        }
        float F2 = F * F;
        float poly_2_pow_F = (1.0f - LN2_F) * F2 + LN2_F * F + 1.0f;
        exp_scores_v0[k] = pow2_I * poly_2_pow_F;
      }
    }
    aie::vector<float, VEC_SIZE> exp_scores_v1;
    for (int k = 0; k < VEC_SIZE; ++k) {
      if (scores_v1[k] < -80.0f) {
        exp_scores_v1[k] = 0.0f;
      } else {
        float y = scores_v1[k] * LOG2E_F;
        int I = static_cast<int>(y);
        if (y < 0.0f && y != static_cast<float>(I)) {
          I--;
        }
        float F = y - static_cast<float>(I);
        float pow2_I;
        if (I < -127) {
          pow2_I = 0.0f; // Below table range, effectively zero
        } else if (I > 0) {
          pow2_I = data[127]; // Index 127 corresponds to I=0
        } else {              // I is in [-127, 0]
          pow2_I = data[I + 127];
        }
        float F2 = F * F;
        float poly_2_pow_F = (1.0f - LN2_F) * F2 + LN2_F * F + 1.0f;
        exp_scores_v1[k] = pow2_I * poly_2_pow_F;
      }
    }
    // --- Sum up the exp values ---
    float sum_exp = aie::reduce_add(exp_scores_v0);
    sum_exp += aie::reduce_add(exp_scores_v1);
    aie::vector<float, VEC_SIZE> normalize_vec =
        aie::broadcast<float, VEC_SIZE>(1.0f / sum_exp);

    aie::vector<float, VEC_SIZE> result_v0 =
        aie::mul(exp_scores_v0, normalize_vec);
    aie::vector<float, VEC_SIZE> result_v1 =
        aie::mul(exp_scores_v1, normalize_vec);
    aie::store_v(current_attn_weights_row_ptr, result_v0);
    aie::store_v(current_attn_weights_row_ptr + VEC_SIZE, result_v1);
  }
}

} // extern "C"