/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

extern "C" {

void vadd(int *A, int *B, int *C) {
  for (int i = 0; i < 32; ++i) {
    C[i] = A[i] + B[i];
  }
}

} // extern "C"
