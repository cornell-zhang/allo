/*
 * Generated VLIW C code for AMD AIE processors
 * Function: simple_add
 */
#define NOCPP
#include <aie_api/aie.hpp>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

extern "C" {

using namespace std;
void simple_add(
  float v0[16],
  float v1[16],
  float v2[16]
) {	// L2
  for (int v3 = 0; v3 < 16; v3++) {	// L5
    v2[v3] = (float)0.000000;	// L5
  }
  l_S_i_0_i: for (int i = 0; i < 16; i++) {	// L6
    float v5 = v0[i];	// L7
    float v6 = v1[i];	// L8
    float v7 = v5 + v6;	// L9
    v2[i] = v7;	// L10
  }
}


} // extern "C"
