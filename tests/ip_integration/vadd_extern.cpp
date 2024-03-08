#include <iostream>
#include "vadd_extern.h"
using namespace std;

extern "C" {

void vadd(int *A, int *B, int *C) {
  for (int i = 0; i < 32; ++i) {
    C[i] = A[i] + B[i];
  }
}

} // extern "C"
