/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <aie_api/aie.hpp>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>
#define NOCPP

// Constants for exp(x) = 2^(x * log2(e))
// log2(e) = 1 / ln(2) = 1.4426950408889634
// ln(2) = 0.6931471805599453
constexpr float LOG2E_F = 1.4426950408889634f;
constexpr float LN2_F = 0.6931471805599453f;

const float lut_data[] = {
    5.87747e-39f, 1.17549e-38f, 2.35099e-38f, 4.70198e-38f, 9.40395e-38f,
    1.88079e-37f, 3.76158e-37f, 7.52316e-37f, 1.50463e-36f, 3.00927e-36f,
    6.01853e-36f, 1.20371e-35f, 2.40741e-35f, 4.81482e-35f, 9.62965e-35f,
    1.92593e-34f, 3.85186e-34f, 7.70372e-34f, 1.54074e-33f, 3.08149e-33f,
    6.16298e-33f, 1.23260e-32f, 2.46519e-32f, 4.93038e-32f, 9.86076e-32f,
    1.97215e-31f, 3.94430e-31f, 7.88861e-31f, 1.57772e-30f, 3.15544e-30f,
    6.31089e-30f, 1.26218e-29f, 2.52435e-29f, 5.04871e-29f, 1.00974e-28f,
    2.01948e-28f, 4.03897e-28f, 8.07794e-28f, 1.61559e-27f, 3.23117e-27f,
    6.46235e-27f, 1.29247e-26f, 2.58494e-26f, 5.16988e-26f, 1.03398e-25f,
    2.06795e-25f, 4.13590e-25f, 8.27181e-25f, 1.65436e-24f, 3.30872e-24f,
    6.61744e-24f, 1.32349e-23f, 2.64698e-23f, 5.29396e-23f, 1.05879e-22f,
    2.11758e-22f, 4.23516e-22f, 8.47033e-22f, 1.69407e-21f, 3.38813e-21f,
    6.77626e-21f, 1.35525e-20f, 2.71051e-20f, 5.42101e-20f, 1.08420e-19f,
    2.16840e-19f, 4.33681e-19f, 8.67362e-19f, 1.73472e-18f, 3.46945e-18f,
    6.93889e-18f, 1.38778e-17f, 2.77556e-17f, 5.55112e-17f, 1.11022e-16f,
    2.22045e-16f, 4.44089e-16f, 8.88178e-16f, 1.77636e-15f, 3.55271e-15f,
    7.10543e-15f, 1.42109e-14f, 2.84217e-14f, 5.68434e-14f, 1.13687e-13f,
    2.27374e-13f, 4.54747e-13f, 9.09495e-13f, 1.81899e-12f, 3.63798e-12f,
    7.27596e-12f, 1.45519e-11f, 2.91038e-11f, 5.82077e-11f, 1.16415e-10f,
    2.32831e-10f, 4.65661e-10f, 9.31323e-10f, 1.86265e-09f, 3.72529e-09f,
    7.45058e-09f, 1.49012e-08f, 2.98023e-08f, 5.96046e-08f, 1.19209e-07f,
    2.38419e-07f, 4.76837e-07f, 9.53674e-07f, 1.90735e-06f, 3.81470e-06f,
    7.62939e-06f, 1.52588e-05f, 3.05176e-05f, 6.10352e-05f, 1.22070e-04f,
    2.44141e-04f, 4.88281e-04f, 9.76562e-04f, 1.95312e-03f, 3.90625e-03f,
    7.81250e-03f, 1.56250e-02f, 3.12500e-02f, 6.25000e-02f, 1.25000e-01f,
    2.50000e-01f, 5.00000e-01f, 1.00000e+00f};

float get_exp(float x) {
  if (x < -80.0f) {
    return 0.0f;
  } else {
    float y = x * LOG2E_F;
    int I = static_cast<int>(y);
    if (y < 0.0f && y != static_cast<float>(I)) {
      I--;
    }
    float F = y - static_cast<float>(I);
    float pow2_I;
    if (I < -127) {
      pow2_I = 0.0f; // Below table range, effectively zero
    } else if (I > 0) {
      pow2_I = lut_data[127]; // Index 127 corresponds to I=0
    } else {                  // I is in [-127, 0]
      pow2_I = lut_data[I + 127];
    }
    float F2 = F * F;
    float poly_2_pow_F = (1.0f - LN2_F) * F2 + LN2_F * F + 1.0f;
    return float(pow2_I * poly_2_pow_F);
  }
}

extern "C" {

void masked_softmax_float32(float attention_score[32][64],
                            int tile_row_start[1], float attn_weights[32][64]) {
  // Define constants for tile dimensions and vectorization
  constexpr int TILE_ROWS = 32;
  constexpr int SEQ_COLS = 64;
  constexpr int VEC_SIZE = 32;

  // Define negative infinity constant
  const float neg_inf = -std::numeric_limits<float>::infinity();
  // Loop over each row in the tile
  for (int r = 0; r < TILE_ROWS; ++r) {
    // Calculate global row index for causal masking
    int global_row_idx = tile_row_start[0] + r;

    // Pointers for current row's input and output
    float *__restrict current_attention_score_row_ptr = &attention_score[r][0];
    float *__restrict current_attn_weights_row_ptr = &attn_weights[r][0];

    for (int k = 0; k < SEQ_COLS; ++k) {
      if (k > global_row_idx) {
        attention_score[r][k] = neg_inf;
      }
    }
    aie::vector<float, VEC_SIZE> scores_v0 =
        aie::load_v<VEC_SIZE>(current_attention_score_row_ptr);
    aie::vector<float, VEC_SIZE> scores_v1 =
        aie::load_v<VEC_SIZE>(current_attention_score_row_ptr + VEC_SIZE);

    // --- Find Max Value for Numerical Stability (LogSumExp trick) ---
    float row_max = aie::reduce_max(scores_v0);
    row_max = std::max(row_max, aie::reduce_max(scores_v1));
    scores_v0 = aie::add(scores_v0, -row_max);
    scores_v1 = aie::add(scores_v1, -row_max);
    // --- Compute exp(x - max) using scalar approximation ---
    float sum_exp = 0.0f;
    for (int k = 0; k < VEC_SIZE; ++k) {
      float exp_value = get_exp(scores_v0[k]);
      sum_exp += exp_value;
      attn_weights[r][k] = exp_value;
    }
    for (int k = 0; k < VEC_SIZE; ++k) {
      float exp_value = get_exp(scores_v1[k]);
      sum_exp += exp_value;
      attn_weights[r][VEC_SIZE + k] = exp_value;
    }
    float scale = 1.0f / sum_exp;
    aie::vector<float, VEC_SIZE> weight_v0 =
        aie::load_v<VEC_SIZE>(current_attn_weights_row_ptr);
    aie::vector<float, VEC_SIZE> weight_v1 =
        aie::load_v<VEC_SIZE>(current_attn_weights_row_ptr + VEC_SIZE);
    weight_v0 = aie::mul(weight_v0, scale);
    weight_v1 = aie::mul(weight_v1, scale);
    aie::store_v(current_attn_weights_row_ptr, weight_v0);
    aie::store_v(current_attn_weights_row_ptr + VEC_SIZE, weight_v1);
  }
}

} // extern "C"