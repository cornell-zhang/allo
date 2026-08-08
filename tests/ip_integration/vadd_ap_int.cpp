/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <ap_int.h>

void vadd_ap_int(ap_int<8> A[32], ap_int<8> B[32], ap_int<16> C[32]) {
  for (int i = 0; i < 32; ++i)
    C[i] = A[i] + B[i];
}
