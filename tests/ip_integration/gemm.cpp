/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//

#include "gemm.h"
using namespace std;

void gemm(
  float v0[16][16],
  float v1[16][16],
  float v2[16][16]
) {	//
  for (int v3 = 0; v3 < 16; v3++) {	//
    for (int v4 = 0; v4 < 16; v4++) {	//
      v2[v3][v4] = 0.000000;	//
    }
  }
  l_S_i_j_0_i: for (int i = 0; i < 16; i++) {	//
    float v6[16];	//
    l_j_init: for (int j_init = 0; j_init < 16; j_init++) {	//
      v6[j_init] = 0.000000;	//
    }
    l_S_k_0_k: for (int k = 0; k < 16; k++) {	//
      l_j: for (int j = 0; j < 16; j++) {	//
        float v10 = v0[i][k];	//
        float v11 = v1[k][j];	//
        float v12 = v10 * v11;	//
        float v13 = v6[j];	//
        float v14 = v13 + v12;	//
        v6[j] = v14;	//
      }
    }
    l_j_back: for (int j_back = 0; j_back < 16; j_back++) {	//
      float v16 = v6[j_back];	//
      v2[i][j_back] = v16;	//
    }
  }
}

