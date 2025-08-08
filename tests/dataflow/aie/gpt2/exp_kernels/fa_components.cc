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

template <typename T_in, typename T_out, const int M, const int N, const int K,
          const float scale>
void for_qk(T_in *tensor_a, T_in *tensor_b, T_in *max_logit,
            T_out *output_tensor, T_out *new_max_logit) {
  constexpr int vec_factor = 16;
  using vec_t = aie::vector<T_in, vec_factor>;
  T_in *__restrict tensor_a_ptr = tensor_a;
  // logits = Q_chunk @ K_chunk.T / np.sqrt(D)
  for (int outer_iter = 0; outer_iter < M; outer_iter++) {
    T_in *__restrict tensor_b_ptr = tensor_b;
    for (int inner_iter = 0; inner_iter < N; ++inner_iter) {
      float sum = 0.0f;
      const int F = K / vec_factor;
      T_in *__restrict tensor_a_tile_ptr = tensor_a_ptr;
      T_in *__restrict tensor_b_tile_ptr = tensor_b_ptr;
      for (int i = 0; i < F; i++) {
        vec_t input_vec_a = aie::load_v<vec_factor>(tensor_a_tile_ptr);
        vec_t input_vec_b = aie::load_v<vec_factor>(tensor_b_tile_ptr);
        tensor_a_tile_ptr += vec_factor;
        tensor_b_tile_ptr += vec_factor;
        vec_t mul_vec = aie::mul(input_vec_a, input_vec_b);
        sum += aie::reduce_add(mul_vec);
      }
      output_tensor[outer_iter * N + inner_iter] =
          static_cast<T_out>(sum * scale);
      tensor_b_ptr += K;
    }
    tensor_a_ptr += K;
  }
  // local_max = np.max(logits, axis=1, keepdims=True)
  // max_logit = np.maximum(max_logit, local_max)
  for (int outer_iter = 0; outer_iter < M; outer_iter++) {
    T_in row_max = max_logit[outer_iter];
    T_in *__restrict logits_ptr = output_tensor + outer_iter * N;
    const int F = N / vec_factor;
    for (int i = 0; i < F; i++) {
      vec_t logits_vec = aie::load_v<vec_factor>(logits_ptr);
      row_max = std::max(row_max, aie::reduce_max(logits_vec));
      logits_ptr += vec_factor;
    }
    new_max_logit[outer_iter] = row_max;
    // logits - max_logit
    logits_ptr = output_tensor + outer_iter * N;
    for (int i = 0; i < F; i++) {
      vec_t logits_vec = aie::load_v<vec_factor>(logits_ptr);
      logits_vec = aie::add(logits_vec, -row_max);
      aie::store_v(logits_ptr, logits_vec);
      logits_ptr += vec_factor;
    }
  }
}

template <typename T_in, typename T_out, const int M, const int N, const int K>
void for_v(T_in *tensor_a, T_in *tensor_b, T_out *output_tensor,
           T_out *sum_exp) {
  // exp_logits = np.exp(tensor_a)

}

extern "C" {

void operand_qk(float A_in[32][64], float B_in[32][64], float max_logit[32],
                float C_out[32][32], float new_max_logit[32]) {
  for_qk<float, float, 32, 32, 64, 0.125f>(&A_in[0][0], &B_in[0][0], max_logit,
                                           &C_out[0][0], new_max_logit);
}

void operand_v(float A_in[32][32], float B_in[32][64], float C_out[32][64],
               float sum_exp[32]) {
  for_qk<float, float, 32, 32, 64>(&A_in[0][0], &B_in[0][0], &C_out[0][0],
                                   sum_exp);
}

} // extern "C"