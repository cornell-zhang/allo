module {
  func.func @matmul_kernel(%arg0: memref<512x512xi32>, %arg1: memref<512x512xi32>) -> memref<512x512xi32> attributes {itypes = "ss", otypes = "s"} {
    %alloc = memref.alloc() : memref<512x512xi32>
    %c0_i32 = arith.constant 0 : i32
    linalg.fill {op_name = "matmul_init_zero_0"} ins(%c0_i32 : i32) outs(%alloc : memref<512x512xi32>)
    linalg.matmul {cast = #linalg.type_fn<cast_signed>, op_name = "matmul_1"} ins(%arg0, %arg1 : memref<512x512xi32>, memref<512x512xi32>) outs(%alloc : memref<512x512xi32>)
    return %alloc : memref<512x512xi32>
  }
}
