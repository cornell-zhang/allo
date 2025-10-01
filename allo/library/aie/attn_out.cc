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

template <typename T_in, typename T_out, int M, int N>
void row_scale(T_in *__restrict tensor_in, T_in *__restrict scale_factors,
               T_out *__restrict tensor_out) {
  constexpr int vec_factor = 256 / (sizeof(T_in) * 8);
  const int F = N / vec_factor;
  for (int outer_iter = 0; outer_iter < M; outer_iter++) {
    T_in *input_ptr = tensor_in + outer_iter * N;
    T_out *output_ptr = tensor_out + outer_iter * N;
    T_in scale_factor = 1.0f / scale_factors[outer_iter];
    for (int i = 0; i < F; i++) {
      aie::vector<T_in, vec_factor> input_vec =
          aie::load_v<vec_factor>(input_ptr);
      aie::vector<T_out, vec_factor> scaled_vec =
          aie::mul(input_vec, scale_factor);
      aie::store_v(output_ptr, scaled_vec);
      input_ptr += vec_factor;
      output_ptr += vec_factor;
    }
  }
}

template <typename T_in, typename T_out, int M, int N>
void row_rescale(T_in *__restrict tensor_in, T_in *__restrict scale_factors,
               T_out *__restrict tensor_out) {
  constexpr int vec_factor = 256 / (sizeof(T_in) * 8);
  const int F = N / vec_factor;
  for (int outer_iter = 0; outer_iter < M; outer_iter++) {
    T_in *input_ptr = tensor_in + outer_iter * N;
    T_out *output_ptr = tensor_out + outer_iter * N;
    T_in scale_factor = scale_factors[outer_iter];
    for (int i = 0; i < F; i++) {
      aie::vector<T_in, vec_factor> input_vec =
          aie::load_v<vec_factor>(input_ptr);
      aie::vector<T_out, vec_factor> scaled_vec =
          aie::mul(input_vec, scale_factor);
      aie::store_v(output_ptr, scaled_vec);
      input_ptr += vec_factor;
      output_ptr += vec_factor;
    }
  }
}

extern "C" {
void scale_attn_output(bfloat16 tensor_in[32][64], bfloat16 sum_exp[32],
                       bfloat16 tensor_out[32][64]) {
  row_scale<bfloat16, bfloat16, 32, 64>(&tensor_in[0][0], sum_exp, &tensor_out[0][0]);
}

void rescale_attn_output(bfloat16 tensor_in[32][64], bfloat16 scale_exp[32],
                       bfloat16 tensor_out[32][64]) {
  row_rescale<bfloat16, bfloat16, 32, 64>(&tensor_in[0][0], scale_exp, &tensor_out[0][0]);
}
}
