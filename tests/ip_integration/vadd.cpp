#include <iostream>
#include "vadd.h"
using namespace std;

void vadd(int A[32]) {
  for (int i = 0; i < 32; ++i) {
    A[i] += 1;
  }
}
