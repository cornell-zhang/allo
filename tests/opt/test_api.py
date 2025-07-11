import allo
from allo._mlir.dialects import func
from allo._mlir.dialects import allo as allo_d
from allo._mlir.ir import *

test_mlir_program = """
module {
func.func private @matmul_scalar_i16_i16(memref<32x128xi16>, memref<128x32xi16>, memref<32x32xi16>) attributes {lib_kernel_name = "matmul"}
func.func private @add_i16_vector(memref<32x32xi16>, memref<32x32xi16>, memref<32x32xi16>) attributes {lib_kernel_name = "add"}
func.func @"gemm_0_0_0-gemm_1_0_0-gemm_2_0_0-gemm_3_0_0"(%arg0: memref<32x128xi16>, %arg1: memref<128x32xi16>, %arg2: memref<32x32xi16>) attributes {df.kernel} {
    %c0_i16 = arith.constant 0 : i16
    linalg.fill ins(%c0_i16 : i16) outs(%arg2 : memref<32x32xi16>)

    %alloc_0 = memref.alloc() : memref<32x32xi16>
    linalg.fill {op_name = "matmul_init_zero_0"} ins(%c0_i16 : i16) outs(%alloc_0 : memref<32x32xi16>)
    call @matmul_scalar_i16_i16(%arg0, %arg1, %alloc_0) : (memref<32x128xi16>, memref<128x32xi16>, memref<32x32xi16>) -> ()
    %alloc_1 = memref.alloc() : memref<32x32xi16>
    call @add_i16_vector(%alloc_0, %arg2, %alloc_1) : (memref<32x32xi16>, memref<32x32xi16>, memref<32x32xi16>) -> ()
    
    %alloc_2 = memref.alloc() {name = "C_out"} : memref<32x32xi16>
    linalg.copy {cast = #linalg.type_fn<cast_signed>} ins(%alloc_1 : memref<32x32xi16>) outs(%alloc_2 : memref<32x32xi16>)
    memref.copy %alloc_2, %arg2 {to = "C"} : memref<32x32xi16> to memref<32x32xi16>
    return
}
}
"""

mod = allo.invoke_mlir_parser(test_mlir_program)
for func_ in mod.body.operations:
    if isinstance(func_, func.FuncOp) and "df.kernel" in func_.attributes:
        for op in func_.entry_block.operations:
            if len(op.operands) > 1:
                print(op)
                v_f = allo_d.get_first_use_in_function(op.operands[1], func_)
                v_l = allo_d.get_last_use_in_function(op.operands[1], func_)
                print("first:", v_f)
                print("last:", v_l)
                print()
