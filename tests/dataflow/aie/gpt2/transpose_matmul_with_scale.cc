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

template <typename T_in, typename T_out, const int M, const int N, const int K,
          const float scale>
void transpose_matmul_with_scale(T_in *tensor_a, T_in *tensor_b,
                                 T_out *output_tensor) {
  constexpr int vec_factor = 16;
  using vec_t = aie::vector<T_in, vec_factor>;
  T_in *__restrict tensor_a_ptr = tensor_a;
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
}

extern "C" {

void transpose_matmul_with_scale(float A_in[32][64], float B_in[32][64],
                        float C_out[32][32]) {
  transpose_matmul_with_scale<float, float, 32, 32, 64, 0.125f>(
      &A_in[0][0], &B_in[0][0], &C_out[0][0]);
}

} // extern "C"