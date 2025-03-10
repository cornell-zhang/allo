/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

void vadd_4d(float A[4][4][16][16], float B[4][4][16][16],
             float C[4][4][16][16]) {
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      for (int k = 0; k < 16; ++k) {
        for (int l = 0; l < 16; ++l) {
          C[i][j][k][l] = A[i][j][k][l] + B[i][j][k][l];
        }
      }
    }
  }
}
