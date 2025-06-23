#include <aie_api/aie.hpp>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define NOCPP

template <typename T, const int BATCH_SIZE, const int FEATURE_DIM>
void gelu_tanh(T *input_tensor, T *output_tensor) {
  constexpr int vec_factor = 16;
  using vec_t = aie::vector<T, vec_factor>;
  event0();
  for (int iter = 0; iter < BATCH_SIZE; iter++) {
    T *__restrict input_ptr = input_tensor + iter * FEATURE_DIM;
    T *__restrict output_ptr = output_tensor + iter * FEATURE_DIM;
    const int F = FEATURE_DIM / vec_factor;
    for (int i = 0; i < F; i++) {
      vec_t input_vec = aie::load_v<vec_factor>(input_ptr);
      input_ptr += vec_factor;
      vec_t inner = aie::mul(vec_t(aie::mul(input_vec, input_vec)),
                             vec_t(aie::mul(input_vec, T(0.044715f))));
      inner = aie::mul(aie::add(input_vec, inner), T(0.797885f));
      // auto tanh_result = aie::tanh(inner); // [NOTE]: require XDNA 2
      // tanh ~ x∗(27+x^2)/(27+9x^2)
      vec_t clamped_inner = aie::min(aie::max(input_vec, T(-2.8f)), T(2.8f));
      vec_t pow2 = aie::mul(clamped_inner, clamped_inner);
      vec_t div =
          aie::div(vec_t(aie::add(pow2, T(27.0f))),
                   vec_t(aie::add(vec_t(aie::mul(pow2, T(9.0f))), T(27.0f))));
      vec_t tanh_result = aie::mul(clamped_inner, div);
      vec_t result = aie::mul(vec_t(aie::mul(input_vec, T(0.5f))),
                              vec_t(aie::add(tanh_result, T(1.0f))));
      aie::store_v(output_ptr, result);
      output_ptr += vec_factor;
    }
  }
  event1();
}

extern "C" {

void gelu(float A_in[4][512], float B_out[4][512]) {
  gelu_tanh<float, 4, 512>(&A_in[0][0], &B_out[0][0]);
}

} // extern "C"