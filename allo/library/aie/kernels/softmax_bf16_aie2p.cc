/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
 
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

template <int L>
void init_softmax(bfloat16 *__restrict max_logit,
                  bfloat16 *__restrict sum_exp) {
  // max_logit = np.full((L, 1), -np.inf)
  // sum_exp = np.zeros((L, 1))
  constexpr int vec_factor =
      256 / (sizeof(bfloat16) * 8); // one 256 bit store unit
  static_assert(L % vec_factor == 0);
  const bfloat16 neg_inf = bfloat16(-std::numeric_limits<float>::infinity());
  const aie::vector<bfloat16, vec_factor> neg_infs =
      aie::broadcast<bfloat16, vec_factor>(neg_inf);
  const aie::vector<bfloat16, vec_factor> zeros =
      aie::zeros<bfloat16, vec_factor>();
  for (int iter = 0; iter < L; iter += vec_factor) {
    aie::store_v(max_logit, neg_infs);
    max_logit += vec_factor;
    aie::store_v(sum_exp, zeros);
    sum_exp += vec_factor;
  }
}

extern "C" {

void exp_bf16(bfloat16 a_in[1024], bfloat16 c_out[1024]) {
  exp_bf16_func<1024>(a_in, c_out);
}

void vector_softmax_bf16(bfloat16 a_in[1024], bfloat16 c_out[1024]) {
  softmax_simple_bf16<1024>(a_in, c_out);
}

void init_softmax(bfloat16 max_logit[32], bfloat16 sum_exp[32]) {
  init_softmax<32>(max_logit, sum_exp);
}

void online_softmax(bfloat16 attention_score[32][32],
                    bfloat16 prev_max_logit[32], bfloat16 prev_sum_exp[32],
                    bfloat16 attention_weight[32][32], bfloat16 scale_exp[32],
                    bfloat16 new_max_logit[32], bfloat16 new_sum_exp[32]) {
  constexpr int ROW = 32;
  constexpr int COL = 32; // == VEC_LEN, one row per vector
  const bfloat16 scale = bfloat16(0.125f);
  aie::vector<bfloat16, VEC_LEN> log2e_vec =
      aie::broadcast<bfloat16, VEC_LEN>((bfloat16)log2e);

  alignas(aie::vector_decl_align) bfloat16 tmp_max_logit[ROW];

  // Pass 1: row max + write (scores * 0.125 - row_max) back into
  // attention_score
  for (int r = 0; r < ROW; r++) {
    bfloat16 *row_ptr = &attention_score[r][0];
    aie::vector<bfloat16, COL> scores = aie::load_v<COL>(row_ptr);
    aie::accum<accfloat, COL> scaled = aie::mul(scores, scale);
    aie::vector<bfloat16, COL> scaled_vec =
        scaled.template to_vector<bfloat16>();
    bfloat16 row_max = std::max(prev_max_logit[r], aie::reduce_max(scaled_vec));

    tmp_max_logit[r] = bfloat16(prev_max_logit[r] - row_max);
    new_max_logit[r] = row_max;

    aie::vector<bfloat16, COL> row_max_vec =
        aie::broadcast<bfloat16, COL>(row_max);
    aie::accum<accfloat, COL> shifted = aie::sub(scaled, row_max_vec);
    aie::store_v(row_ptr, shifted.template to_vector<bfloat16>());
  }

  // Pass 2: scale_exp[0..32) = exp(tmp_max_logit) via 2^(log2e * x)
  {
    aie::vector<bfloat16, COL> v = aie::load_v<COL>(&tmp_max_logit[0]);
    aie::accum<accfloat, COL> z = aie::mul(v, log2e_vec);
    aie::vector<bfloat16, COL> e =
        aie::exp2<bfloat16>(z.template to_vector<float>());
    aie::store_v(&scale_exp[0], e);
  }

  // Pass 3: per-row exp, sum, and new_sum_exp update
  for (int r = 0; r < ROW; r++) {
    aie::vector<bfloat16, COL> shifted =
        aie::load_v<COL>(&attention_score[r][0]);
    aie::accum<accfloat, COL> z = aie::mul(shifted, log2e_vec);
    aie::vector<bfloat16, COL> exp_val =
        aie::exp2<bfloat16>(z.template to_vector<float>());
    aie::store_v(&attention_weight[r][0], exp_val);

    aie::accum<accfloat, COL> sum_acc;
    sum_acc.from_vector(exp_val, 0);
    float accum_exp_val = aie::reduce_add(sum_acc.template to_vector<float>());

    new_sum_exp[r] =
        bfloat16(prev_sum_exp[r] * scale_exp[r] + bfloat16(accum_exp_val));
  }
}

} // extern "C"