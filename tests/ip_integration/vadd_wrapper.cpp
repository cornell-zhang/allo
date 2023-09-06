#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "vadd.h"

extern "C" void pyvadd(int64_t rank, void *ptr) {
  UnrankedMemRefType<int> inA = {rank, ptr};
  DynamicMemRefType<int> A(inA);
  int *A_ptr = (int *)A.data;
  vadd(A_ptr);
}
