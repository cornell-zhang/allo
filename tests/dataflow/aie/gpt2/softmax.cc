/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <aie_api/aie.hpp>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define NOCPP

// softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))

template <typename T, const int BATCH_SIZE, const int SEQUENCE_LEN>
void safe_softmax(T *input_tensor, T *output_tensor) {
  constexpr int vec_factor = 16;
  using vec_t = aie::vector<T, vec_factor>;
  // exp(x) ≈ a3*x^3 + a2*x^2 + a1*x + a0
  constexpr float a0 = 1.00001f;
  constexpr float a1 = 0.241081f;
  constexpr float a2 = 0.0523671f;
  constexpr float a3 = 0.00313326f;
  for (int iter = 0; iter < BATCH_SIZE; iter++) {
    T *input_ptr = input_tensor + iter * SEQUENCE_LEN;
    T *output_ptr = output_tensor + iter * SEQUENCE_LEN;
    T max_value = T(-10000.0f);
    const int F = SEQUENCE_LEN / vec_factor;
    for (int i = 0; i < F; i++) {
      vec_t input_vec = aie::load_v<vec_factor>(input_ptr);
      input_ptr += vec_factor;
      T vec_max_value = aie::reduce_max(input_vec);
      if (vec_max_value > max_value) {
        max_value = vec_max_value;
      }
    }
    vec_t max_vec = aie::broadcast<T, vec_factor>(max_value);
    input_ptr = input_tensor + iter * SEQUENCE_LEN;
    float sum = 0.0f;
    for (int i = 0; i < F; i++) {
      vec_t input_vec = aie::load_v<vec_factor>(input_ptr);
      input_ptr += vec_factor;
      vec_t sub_vec = aie::sub(input_vec, max_vec);
      // exp(x) ≈ a3*x^3 + a2*x^2 + a1*x + a0
      vec_t pow2 = aie::mul(sub_vec, sub_vec);
      vec_t pow3 = aie::mul(pow2, sub_vec);
      pow3 = aie::mul(pow3, T(a3));
      pow2 = aie::mul(pow3, T(a2));
      vec_t exp_vec = aie::add(vec_t(aie::add(pow3, pow2)),
                               vec_t(aie::mul(sub_vec, T(a1))));
      exp_vec = aie::add(exp_vec, T(a0));
      aie::store_v(output_ptr, exp_vec);
      output_ptr += vec_factor;
      sum += aie::reduce_add(exp_vec);
    }
    output_ptr = output_tensor + iter * SEQUENCE_LEN;
    for (int i = 0; i < F; i++) {
      vec_t exp_vec = aie::load_v<vec_factor>(output_ptr);
      vec_t div = aie::div(exp_vec, sum);
      aie::store_v(output_ptr, div);
      output_ptr += vec_factor;
    }
  }
}

extern "C" {

void softmax(float A_in[1][32], float B_out[1][32]) {
  safe_softmax<float, 1, 32>(&A_in[0][0], &B_out[0][0]);
}

} // extern "C"