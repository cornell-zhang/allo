/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

void vadd_int(int A[32], int B[32], int val) {
  for (int i = 0; i < 32; ++i) {
    B[i] = A[i] + val;
  }
}
