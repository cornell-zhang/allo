/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
 
//===- gemm.cc -------------------------------------------------*- C++ -*-===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>


extern "C" {

void strange_kernel_int16(int16_t A[64][64], int16_t B[64][64], int16_t C[64][64]) {
  event0();
  
  // Direct implementation without templates
  int16_t *a = reinterpret_cast<int16_t*>(A);
  int16_t *b = reinterpret_cast<int16_t*>(B);
  int16_t *c = reinterpret_cast<int16_t*>(C);
  
  for (int i = 0; i < 64; i++) {
    for (int j = 0; j < 64; j++) {
      int16_t sum = 0;
      int16_t *a_row = a + i * 64;  // Row i of matrix A
      
      // Simple dot product - compute row i of A with column j of B
      for (int k = 0; k < 64; k++)
        chess_prepare_for_pipelining
        chess_loop_range(64, )
        {
          sum += a_row[k] * b[k * 64 + j];
        }
      
      c[i * 64 + j] = sum;
    }
  }
  
  event1();
}

} // extern "C" 
