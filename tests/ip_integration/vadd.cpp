#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include <iostream>
using namespace std;

extern "C" void vadd(int64_t rank, void *ptr) { //
  std::cout << "Enter vadd function" << std::endl;
  UnrankedMemRefType<int> inA = {rank, ptr};
  DynamicMemRefType<int> A(inA);
  int *A_ptr = (int*) A.data;
  for (int i = 0; i < 8; ++i) {
    std::cout << A_ptr[i] << " ";
  }
  std::cout << std::endl;
}