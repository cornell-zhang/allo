//===- bf16_exp.cc ---------------------------*- C++-----*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===-----------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

#define VEC_LEN 32
#define log2e 1.44269504089

using namespace aie;

template <const int N>
void exp_bf16_func(bfloat16 *restrict in, bfloat16 *restrict out) {
  auto it_exp_in = aie::cbegin_vector<VEC_LEN>((bfloat16 *)in);
  auto it_exp_out = aie::begin_vector<VEC_LEN>((bfloat16 *)out);

  const int elem_iters = N / VEC_LEN;

  // Calculate the e^(x) function as 2^(log2e * x)
  aie::vector<bfloat16, VEC_LEN> input_bf16;
  aie::accum<accfloat, VEC_LEN> exp_in;
  aie::vector<bfloat16, VEC_LEN> exp_val;
  aie::vector<bfloat16, VEC_LEN> log2e_vec =
      aie::broadcast<bfloat16, VEC_LEN>(log2e);

  for (int i = 0; i < elem_iters; i++) {
    input_bf16 = *it_exp_in++;
    exp_in = aie::mul(input_bf16, log2e_vec);
    exp_val = aie::exp2<bfloat16>(exp_in.to_vector<float>());
    *it_exp_out++ = exp_val;
  }
}

template <const int N>
void softmax_simple_bf16(bfloat16 *restrict input_vector,
                         bfloat16 *restrict output_vector) {
  event0();
  // VJUNG: We do 3 passes on the vector:
  // 1. Find the max value scaled by log2e in the vector
  // 2. Calculate the exponentials of the scaled values minus the maximum
  // 3. Calculate the softmax by dividing each exponential by the sum of all
  // exponentials Note: The multiplication by log2e is very sensitive, casting
  // it to bf16 before exponentiation leads to wrong output.
  auto it_log_in =
      aie::cbegin_restrict_vector<VEC_LEN>((bfloat16 *)input_vector);
  auto it_log_out =
      aie::begin_restrict_vector<VEC_LEN>((bfloat16 *)input_vector);
  auto it_exp_in =
      aie::cbegin_restrict_vector<VEC_LEN>((bfloat16 *)input_vector);
  auto it_exp_out =
      aie::begin_restrict_vector<VEC_LEN>((bfloat16 *)output_vector);
  auto it_scale =
      aie::cbegin_restrict_vector<VEC_LEN>((bfloat16 *)output_vector);
  auto it_soft_out =
      aie::begin_restrict_vector<VEC_LEN>((bfloat16 *)output_vector);

  aie::vector<bfloat16, VEC_LEN> in_elems, exp_val, input_bf16, log2e_vec,
      max_val_vec;
  aie::accum<accfloat, VEC_LEN> out_vals, exp_val_accum, scaled_accum,
      exp_in_accum;

  float max_val = 0;
  float accum_exp_val = 0;
  float running_max = 0;
  bfloat16 col_sum_inv;
  const int elem_iters = N / VEC_LEN;

  exp_val_accum = aie::zeros<accfloat, VEC_LEN>();

  log2e_vec = aie::broadcast<bfloat16, VEC_LEN>((bfloat16)log2e);

  // First pass - Optimized: element-wise max + single final reduce_max
  // Use vector max accumulation, then reduce once at the end
  aie::vector<bfloat16, VEC_LEN> max_accum_vec =
      aie::broadcast<bfloat16, VEC_LEN>((bfloat16)-32768.0f);
  for (int i = 0; i < elem_iters; i++) {
    input_bf16 = *it_log_in++;
    scaled_accum = aie::mul(input_bf16, log2e_vec);
    max_accum_vec = aie::max(max_accum_vec, scaled_accum.to_vector<bfloat16>());
  }
  max_val = aie::reduce_max(max_accum_vec);
  max_val_vec = aie::broadcast<bfloat16, VEC_LEN>(max_val);

  // Second pass
  for (int i = 0; i < elem_iters; i++) {

    input_bf16 = *it_exp_in++;

    scaled_accum = aie::mul(input_bf16, log2e_vec);
    exp_in_accum = aie::sub(scaled_accum, max_val_vec);
    exp_val = aie::exp2<bfloat16>(exp_in_accum.to_vector<float>());
    exp_val_accum = add(exp_val_accum, exp_val);

    *it_exp_out++ = exp_val;
  }

  // Final reduction after loop
  aie::vector<float, VEC_LEN> reduce = exp_val_accum.to_vector<float>();
  accum_exp_val = aie::reduce_add(reduce);
  col_sum_inv = (bfloat16)aie::inv(accum_exp_val);

  for (int c = 0; c < elem_iters; c++) {
    in_elems = *it_scale++;
    out_vals = aie::mul(in_elems, col_sum_inv);
    *it_soft_out++ = out_vals.to_vector<bfloat16>();
  }

  event1();
  return;
}

extern "C" {

void exp_bf16(bfloat16 a_in[1024], bfloat16 c_out[1024]) {
  exp_bf16_func<1024>(a_in, c_out);
}

void vector_softmax_bf16(bfloat16 a_in[1024], bfloat16 c_out[1024]) {
  softmax_simple_bf16<1024>(a_in, c_out);
}

} // extern "C"