# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo._mlir.passmanager import PassManager

test_mlir_program = """
module {
  func.func private @matmul_scalar_i16_i16(memref<32x128xi16>, memref<128x32xi16>, memref<32x32xi16>) attributes {lib_kernel_name = "matmul"}
  func.func private @add_i16_vector(memref<32x32xi16>, memref<32x32xi16>, memref<32x32xi16>) attributes {lib_kernel_name = "add"}
  func.func @"gemm_0_0_0-gemm_1_0_0-gemm_2_0_0-gemm_3_0_0"(%arg0: memref<32x128xi16>, %arg1: memref<128x32xi16>, %arg2: memref<32x32xi16>, %arg3: !allo.stream<memref<32x32xi16>, 2>, %arg4: memref<32x128xi16>, %arg5: memref<128x32xi16>, %arg6: memref<32x32xi16>, %arg7: !allo.stream<memref<32x32xi16>, 2>, %arg8: !allo.stream<memref<32x32xi16>, 2>, %arg9: memref<32x128xi16>, %arg10: memref<128x32xi16>, %arg11: memref<32x32xi16>, %arg12: !allo.stream<memref<32x32xi16>, 2>, %arg13: !allo.stream<memref<32x32xi16>, 2>, %arg14: memref<32x128xi16>, %arg15: memref<128x32xi16>, %arg16: memref<32x32xi16>, %arg17: !allo.stream<memref<32x32xi16>, 2>) attributes {df.kernel} {
    %c0_i16 = arith.constant 0 : i16
    %alloc = memref.alloc() {name = "C_in"} : memref<32x32xi16>
    linalg.fill ins(%c0_i16 : i16) outs(%alloc : memref<32x32xi16>)
    %alloc_0 = memref.alloc() : memref<32x32xi16>
    linalg.fill {op_name = "matmul_init_zero_0"} ins(%c0_i16 : i16) outs(%alloc_0 : memref<32x32xi16>)
    call @matmul_scalar_i16_i16(%arg0, %arg1, %alloc_0) : (memref<32x128xi16>, memref<128x32xi16>, memref<32x32xi16>) -> ()
    %alloc_1 = memref.alloc() : memref<32x32xi16>
    call @add_i16_vector(%alloc_0, %alloc, %alloc_1) : (memref<32x32xi16>, memref<32x32xi16>, memref<32x32xi16>) -> ()
    %alloc_2 = memref.alloc() {name = "C_out"} : memref<32x32xi16>
    linalg.copy {cast = #linalg.type_fn<cast_signed>} ins(%alloc_1 : memref<32x32xi16>) outs(%alloc_2 : memref<32x32xi16>)
    %c0_i16_3 = arith.constant 0 : i16
    %alloc_4 = memref.alloc() : memref<32x32xi16>
    linalg.fill {op_name = "matmul_init_zero_0"} ins(%c0_i16_3 : i16) outs(%alloc_4 : memref<32x32xi16>)
    call @matmul_scalar_i16_i16(%arg4, %arg5, %alloc_4) : (memref<32x128xi16>, memref<128x32xi16>, memref<32x32xi16>) -> ()
    %alloc_5 = memref.alloc() : memref<32x32xi16>
    call @add_i16_vector(%alloc_4, %alloc_2, %alloc_5) : (memref<32x32xi16>, memref<32x32xi16>, memref<32x32xi16>) -> ()
    %alloc_6 = memref.alloc() {name = "C_out"} : memref<32x32xi16>
    linalg.copy {cast = #linalg.type_fn<cast_signed>} ins(%alloc_5 : memref<32x32xi16>) outs(%alloc_6 : memref<32x32xi16>)
    %c0_i16_7 = arith.constant 0 : i16
    %alloc_8 = memref.alloc() : memref<32x32xi16>
    linalg.fill {op_name = "matmul_init_zero_0"} ins(%c0_i16_7 : i16) outs(%alloc_8 : memref<32x32xi16>)
    call @matmul_scalar_i16_i16(%arg9, %arg10, %alloc_8) : (memref<32x128xi16>, memref<128x32xi16>, memref<32x32xi16>) -> ()
    %alloc_9 = memref.alloc() : memref<32x32xi16>
    call @add_i16_vector(%alloc_8, %alloc_6, %alloc_9) : (memref<32x32xi16>, memref<32x32xi16>, memref<32x32xi16>) -> ()
    %alloc_10 = memref.alloc() {name = "C_out"} : memref<32x32xi16>
    linalg.copy {cast = #linalg.type_fn<cast_signed>} ins(%alloc_9 : memref<32x32xi16>) outs(%alloc_10 : memref<32x32xi16>)
    %c0_i16_11 = arith.constant 0 : i16
    %alloc_12 = memref.alloc() : memref<32x32xi16>
    linalg.fill {op_name = "matmul_init_zero_0"} ins(%c0_i16_11 : i16) outs(%alloc_12 : memref<32x32xi16>)
    call @matmul_scalar_i16_i16(%arg14, %arg15, %alloc_12) : (memref<32x128xi16>, memref<128x32xi16>, memref<32x32xi16>) -> ()
    %alloc_13 = memref.alloc() : memref<32x32xi16>
    call @add_i16_vector(%alloc_12, %alloc_10, %alloc_13) : (memref<32x32xi16>, memref<32x32xi16>, memref<32x32xi16>) -> ()
    %alloc_14 = memref.alloc() {name = "C_out"} : memref<32x32xi16>
    linalg.copy {cast = #linalg.type_fn<cast_signed>} ins(%alloc_13 : memref<32x32xi16>) outs(%alloc_14 : memref<32x32xi16>)
    memref.copy %alloc_14, %arg16 {to = "C"} : memref<32x32xi16> to memref<32x32xi16>
    return
  }
}
"""

mod = allo.invoke_mlir_parser(test_mlir_program)

# TODO: new pass
pipeline = f"builtin.module(memref-dce)"
with mod.context:
    PassManager.parse(pipeline).run(mod.operation)

print(mod)
