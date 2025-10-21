/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

void vadd(int A[32], int B[32], int C[32]) {
  for (int i = 0; i < 32; ++i) {
    C[i] = A[i] + B[i];
  }
}
