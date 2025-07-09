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

constexpr float TANH_MAX_X = 4.0f; // beyond this, tanh(x) ≈ ±1
constexpr int TANH_LUT_SIZE = 128; // number of samples on [0, TANH_MAX_X]
constexpr float TANH_STEP = TANH_MAX_X / (TANH_LUT_SIZE - 1);

const float lut_data[] = {
    0.00000000e+00, 3.14856524e-02, 6.29089403e-02, 9.42079922e-02,
    1.25321915e-01, 1.56191261e-01, 1.86758476e-01, 2.16968308e-01,
    2.46768191e-01, 2.76108575e-01, 3.04943216e-01, 3.33229421e-01,
    3.60928233e-01, 3.88004578e-01, 4.14427347e-01, 4.40169442e-01,
    4.65207769e-01, 4.89523194e-01, 5.13100451e-01, 5.35928024e-01,
    5.57997994e-01, 5.79305862e-01, 5.99850350e-01, 6.19633189e-01,
    6.38658891e-01, 6.56934515e-01, 6.74469434e-01, 6.91275092e-01,
    7.07364775e-01, 7.22753379e-01, 7.37457189e-01, 7.51493669e-01,
    7.64881260e-01, 7.77639195e-01, 7.89787318e-01, 8.01345926e-01,
    8.12335620e-01, 8.22777169e-01, 8.32691388e-01, 8.42099030e-01,
    8.51020688e-01, 8.59476712e-01, 8.67487135e-01, 8.75071608e-01,
    8.82249349e-01, 8.89039099e-01, 8.95459082e-01, 9.01526981e-01,
    9.07259911e-01, 9.12674407e-01, 9.17786412e-01, 9.22611271e-01,
    9.27163728e-01, 9.31457930e-01, 9.35507431e-01, 9.39325202e-01,
    9.42923636e-01, 9.46314566e-01, 9.49509275e-01, 9.52518511e-01,
    9.55352505e-01, 9.58020985e-01, 9.60533195e-01, 9.62897912e-01,
    9.65123464e-01, 9.67217746e-01, 9.69188242e-01, 9.71042038e-01,
    9.72785841e-01, 9.74425998e-01, 9.75968508e-01, 9.77419046e-01,
    9.78782969e-01, 9.80065340e-01, 9.81270937e-01, 9.82404271e-01,
    9.83469596e-01, 9.84470927e-01, 9.85412049e-01, 9.86296529e-01,
    9.87127729e-01, 9.87908819e-01, 9.88642784e-01, 9.89332434e-01,
    9.89980417e-01, 9.90589225e-01, 9.91161206e-01, 9.91698567e-01,
    9.92203387e-01, 9.92677621e-01, 9.93123108e-01, 9.93541581e-01,
    9.93934666e-01, 9.94303895e-01, 9.94650707e-01, 9.94976457e-01,
    9.95282416e-01, 9.95569783e-01, 9.95839681e-01, 9.96093169e-01,
    9.96331240e-01, 9.96554829e-01, 9.96764813e-01, 9.96962018e-01,
    9.97147220e-01, 9.97321146e-01, 9.97484482e-01, 9.97637871e-01,
    9.97781917e-01, 9.97917188e-01, 9.98044218e-01, 9.98163507e-01,
    9.98275527e-01, 9.98380719e-01, 9.98479499e-01, 9.98572258e-01,
    9.98659362e-01, 9.98741156e-01, 9.98817962e-01, 9.98890084e-01,
    9.98957808e-01, 9.99021402e-01, 9.99081117e-01, 9.99137190e-01,
    9.99189842e-01, 9.99239283e-01, 9.99285707e-01, 9.99329300e-01};

float get_tanh(float x) {
  if (x >= TANH_MAX_X) {
    return 1.0f;
  }
  if (x <= -TANH_MAX_X) {
    return -1.0f;
  }
  // Work in [0, MAX_X] via symmetry
  bool neg = x < 0.0f;
  float ax = neg ? -x : x;

  // Compute floating index
  float idx_f = ax / TANH_STEP;
  int i = static_cast<int>(idx_f);
  float frac = idx_f - i;

  // Gather and linearly interpolate
  float y0 = lut_data[i];
  float y1 = lut_data[i + 1];
  float y = y0 + frac * (y1 - y0);

  return neg ? -y : y;
}

extern "C" {

void gelu_float32(float input_x[4][768], float output_x[4][768]) {
  constexpr int SEQ_TILE = 4;
  constexpr int FEATURE_DIM_TILE = 768;
  constexpr int vec_factor = 16;
  using vec_t = aie::vector<float, vec_factor>;
  for (int iter = 0; iter < SEQ_TILE; iter++) {
    float *__restrict input_ptr = &input_x[iter][0];
    float *__restrict output_ptr = &output_x[iter][0];
    for (int iter = 0; iter < FEATURE_DIM_TILE; iter++) {
      float value = input_ptr[iter];
      float inner = 0.797885f * value * (1 + 0.044715f * value * value);
      output_ptr[iter] = (get_tanh(inner) + 1.0f) * value * 0.5f;
    }
  }
}

} // extern "C"