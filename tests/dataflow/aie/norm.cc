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

#define EPS 1e-6f // epsilon

template <typename T_in, typename T_out, const int SEQ_LEN, const int HIDDEN>
void layer_norm_single_batch_no_bias(T_in *input_tensor, T_in *weight,
                                     T_out *output_tensor) {
  constexpr int vec_factor = 16;
  using vec_t = aie::vector<T_in, vec_factor>;
  event0();
  for (int iter = 0; iter < SEQ_LEN; iter++) {
    T_in *__restrict input_ptr = input_tensor;
    T_in *__restrict weight_ptr = weight;
    T_out *__restrict output_ptr = output_tensor;
    float mean = 0.0f, variance_sum = 0.0f;
    const int F = HIDDEN / vec_factor;
    for (int i = 0; i < F; i++) {
      vec_t input_vec = aie::load_v<vec_factor>(input_ptr);
      input_ptr += vec_factor;
      mean += aie::reduce_add(input_vec);
    }
    mean /= HIDDEN;
    input_ptr = input_tensor;
    for (int i = 0; i < F; i++) {
      vec_t input_vec = aie::load_v<vec_factor>(input_ptr);
      input_ptr += vec_factor;
      vec_t diff = aie::sub(input_vec, mean);
      vec_t square_vec = aie::mul(diff, diff);
      variance_sum += aie::reduce_add(square_vec);
    }
    vec_t variance_vec =
        aie::broadcast<T_in, vec_factor>(variance_sum / HIDDEN + EPS);
    vec_t rms = aie::invsqrt(variance_vec);
    input_ptr = input_tensor;
    for (int i = 0; i < F; i++) {
      vec_t input_vec = aie::load_v<vec_factor>(input_ptr);
      input_ptr += vec_factor;
      vec_t normed = aie::mul(aie::sub(input_vec, mean), rms);
      vec_t weight_vec = aie::load_v<vec_factor>(weight_ptr);
      weight_ptr += vec_factor;
      vec_t result = aie::mul(normed, weight_vec);
      aie::store_v(output_ptr, result);
      output_ptr += vec_factor;
    }
    input_tensor += HIDDEN;
    output_tensor += HIDDEN;
  }
  event1();
}

template <typename T_in, typename T_out, const int SEQ_LEN, const int HIDDEN>
void rms_norm_single_batch(T_in *input_tensor, T_in *weight,
                           T_out *output_tensor) {
  constexpr int vec_factor = 16;
  using vec_t = aie::vector<T_in, vec_factor>;
  event0();
  for (int iter = 0; iter < SEQ_LEN; iter++) {
    T_in *__restrict input_ptr = input_tensor;
    T_in *__restrict weight_ptr = weight;
    T_out *__restrict output_ptr = output_tensor;
    float square_sum = 0.0f;
    const int F = HIDDEN / vec_factor;
    for (int i = 0; i < F; i++) {
      vec_t input_vec = aie::load_v<vec_factor>(input_ptr);
      input_ptr += vec_factor;
      vec_t square_vec = aie::mul(input_vec, input_vec);
      square_sum += aie::reduce_add(square_vec);
    }
    vec_t square_sum_vec =
        aie::broadcast<T_in, vec_factor>(square_sum / HIDDEN + EPS);
    vec_t rms = aie::invsqrt(square_sum_vec);
    input_ptr = input_tensor;
    for (int i = 0; i < F; i++) {
      vec_t input_vec = aie::load_v<vec_factor>(input_ptr);
      input_ptr += vec_factor;
      vec_t normed = aie::mul(input_vec, rms);
      vec_t weight_vec = aie::load_v<vec_factor>(weight_ptr);
      weight_ptr += vec_factor;
      vec_t result = aie::mul(normed, weight_vec);
      aie::store_v(output_ptr, result);
      output_ptr += vec_factor;
    }
    input_tensor += HIDDEN;
    output_tensor += HIDDEN;
  }
  event1();
}

extern "C" {

void layer_norm(float A_in[4][512], float B_in[512], float C_out[4][512]) {
  layer_norm_single_batch_no_bias<float, float, 4, 512>(&A_in[0][0], B_in, &C_out[0][0]);
}

void rms_norm(float A_in[4][512], float B_in[512], float C_out[4][512]) {
  rms_norm_single_batch<float, float, 4, 512>(&A_in[0][0], B_in, &C_out[0][0]);
}

} // extern "C"