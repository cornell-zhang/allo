/*
 * Generated VLIW C code for AMD AIE processors
 * Function: matrix_multiply
 */
#define NOCPP
#include <aie_api/aie.hpp>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

extern "C" {

using namespace std;
void matrix_multiply(
  float v0[4][4],
  float v1[4][4],
  float v2[4][4]
) {	// L2
  for (int v3 = 0; v3 < 4; v3++) {	// L5
    for (int v4 = 0; v4 < 4; v4++) {	// L5
      v2[v3][v4] = (float)0.000000;	// L5
    }
  }
  l_S_i_0_i: for (int i = 0; i < 4; i++) {	// L6
    l_S_j_0_j: for (int j = 0; j < 4; j++) {	// L7
      l_S_k_0_k: for (int k = 0; k < 4; k++) {	// L8
        float v8 = v0[i][k];	// L9
        float v9 = v1[k][j];	// L10
        float v10 = v8 * v9;	// L11
        float v11 = v2[i][j];	// L12
        float v12 = v11 + v10;	// L13
        v2[i][j] = v12;	// L14
      }
    }
  }
}


} // extern "C"
