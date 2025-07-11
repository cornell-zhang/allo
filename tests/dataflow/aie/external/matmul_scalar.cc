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

template <typename T_in, typename T_out, int rowA, int colA, int colB>
static inline void matmul_scalar(T_in *a, T_in *b, T_out *c_in, T_out *c) {
  for (int row = 0; row < rowA; row++) {
    for (int col = 0; col < colB; col++) {
      T_out running_sum = 0;
      for (int i = 0; i < colA; i++) {
        running_sum += a[row * colA + i] * b[i * colB + col];
      }
      c[row * colB + col] = c_in[row * colB + col] + running_sum;
    }
  }
}

extern "C" {

void matmul_scalar_i16_i16(int16_t A_in[32][128], int16_t B_in[128][32],
                           int16_t C_in[32][32], int16_t C_out[32][32]) {
  matmul_scalar<int16_t, int16_t, 32, 128, 32>(&A_in[0][0], &B_in[0][0],
                                               &C_in[0][0], &C_out[0][0]);
}

void matmul_scalar_i16_i32(int16_t A_in[32][128], int16_t B_in[128][32],
                           int32_t C_in[32][32], int32_t C_out[32][32]) {
  matmul_scalar<int16_t, int32_t, 32, 128, 32>(&A_in[0][0], &B_in[0][0],
                                               &C_in[0][0], &C_out[0][0]);
}

} // extern "C"