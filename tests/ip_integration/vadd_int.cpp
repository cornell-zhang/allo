/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <iostream>
#include "vadd_int.h"
using namespace std;

void vadd_int(int A[32], int B[32], int val) {
  for (int i = 0; i < 32; ++i) {
    B[i] = A[i] + val;
  }
}
