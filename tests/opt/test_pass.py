# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo._mlir.passmanager import PassManager

test_mlir_program = """
module {
func.func @matmul(%A: memref<32x32xi32>, %B: memref<32x32xi32>) -> memref<32x32xi32> {
  %C = memref.alloc() : memref<32x32xi32>
  %C2 = memref.alloc() : memref<32x32xi32>
  %c0_i32 = arith.constant 0 : i32
  linalg.fill ins(%c0_i32 : i32) outs(%C : memref<32x32xi32>)
  linalg.matmul ins(%A, %B: memref<32x32xi32>, memref<32x32xi32>)
                outs(%C: memref<32x32xi32>)
  return %C: memref<32x32xi32>
}
}
"""

mod = allo.invoke_mlir_parser(test_mlir_program)

pipeline = f"builtin.module(memref-dce)"
with mod.context:
    PassManager.parse(pipeline).run(mod.operation)

print(mod)
