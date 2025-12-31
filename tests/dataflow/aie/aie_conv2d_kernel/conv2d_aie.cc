/*
 * Generated VLIW C code for AMD AIE processors
 * Function: conv2d
 */
#define NOCPP
#include <aie_api/aie.hpp>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

extern "C" {

using namespace std;
void conv2d(
  float v0[4][4],
  float v1[3][3],
  float v2[2][2]
) {	// L2
  for (int v3 = 0; v3 < 2; v3++) {	// L5
    for (int v4 = 0; v4 < 2; v4++) {	// L5
      v2[v3][v4] = (float)0.000000;	// L5
    }
  }
  l_S_i_0_i: for (int i = 0; i < 2; i++) {	// L6
    l_S_j_0_j: for (int j = 0; j < 2; j++) {	// L7
      float sum_val;	// L10
      sum_val = (float)0.000000;	// L11
      l_S_ki_0_ki: for (int ki = 0; ki < 3; ki++) {	// L12
        l_S_kj_0_kj: for (int kj = 0; kj < 3; kj++) {	// L13
          float v10 = v0[(i + ki)][(j + kj)];	// L14
          float v11 = v1[ki][kj];	// L15
          float v12 = v10 * v11;	// L16
          float v13 = sum_val;	// L17
          float v14 = v13 + v12;	// L18
          sum_val = v14;	// L19
        }
      }
      float v15 = sum_val;	// L22
      v2[i][j] = v15;	// L23
    }
  }
}


} // extern "C"
