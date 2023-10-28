#map = affine_map<()[s0] -> (s0 * 64)>
#map1 = affine_map<()[s0] -> (s0 * 32)>
module {
  func.func @matmul_kernel(%arg0: memref<512x512xi32>, %arg1: memref<512x512xi32>) -> memref<512x512xi32> attributes {itypes = "ss", otypes = "s"} {
    %c8 = arith.constant 8 : index
    %c0_i32 = arith.constant 0 : i32
    %alloc = memref.alloc() : memref<512x512xi32>
    linalg.fill {op_name = "matmul_init_zero_0"} ins(%c0_i32 : i32) outs(%alloc : memref<512x512xi32>)
    air.launch (%arg2, %arg3) in (%arg4=%c8, %arg5=%c8) args(%arg6=%arg0, %arg7=%arg1, %arg8=%alloc) : memref<512x512xi32>, memref<512x512xi32>, memref<512x512xi32> {
      air.segment  args(%arg9=%arg2, %arg10=%arg3, %arg11=%arg6, %arg12=%arg7, %arg13=%arg8) : index, index, memref<512x512xi32>, memref<512x512xi32>, memref<512x512xi32> {
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c0 = arith.constant 0 : index
        %c512 = arith.constant 512 : index
        %c64 = arith.constant 64 : index
        %0 = affine.apply #map()[%arg9]
        %1 = affine.apply #map()[%arg10]
        scf.for %arg14 = %c0 to %c512 step %c64 {
          %alloc_0 = memref.alloc() : memref<64x64xi32, 1>
          %alloc_1 = memref.alloc() : memref<64x64xi32, 1>
          %alloc_2 = memref.alloc() : memref<64x64xi32, 1>
          air.dma_memcpy_nd (%alloc_0[] [] [], %arg11[%0, %arg14] [%c64, %c64] [%c512, %c1]) {id = 1 : i32} : (memref<64x64xi32, 1>, memref<512x512xi32>)
          air.dma_memcpy_nd (%alloc_1[] [] [], %arg12[%arg14, %1] [%c64, %c64] [%c512, %c1]) {id = 2 : i32} : (memref<64x64xi32, 1>, memref<512x512xi32>)
          air.dma_memcpy_nd (%alloc_2[] [] [], %arg13[%0, %1] [%c64, %c64] [%c512, %c1]) {id = 3 : i32} : (memref<64x64xi32, 1>, memref<512x512xi32>)
          air.herd @herd_0  tile (%arg15, %arg16) in (%arg17=%c2, %arg18=%c2) args(%arg19=%alloc_0, %arg20=%alloc_1, %arg21=%alloc_2) : memref<64x64xi32, 1>, memref<64x64xi32, 1>, memref<64x64xi32, 1> {
            %c1_3 = arith.constant 1 : index
            %c0_4 = arith.constant 0 : index
            %c64_5 = arith.constant 64 : index
            %c32 = arith.constant 32 : index
            %2 = affine.apply #map1()[%arg15]
            %3 = affine.apply #map1()[%arg16]
            scf.for %arg22 = %c0_4 to %c64_5 step %c32 {
              %alloc_6 = memref.alloc() : memref<32x32xi32, 2>
              %alloc_7 = memref.alloc() : memref<32x32xi32, 2>
              %alloc_8 = memref.alloc() : memref<32x32xi32, 2>
              air.dma_memcpy_nd (%alloc_6[] [] [], %arg19[%2, %arg22] [%c32, %c32] [%c64_5, %c1_3]) {id = 4 : i32} : (memref<32x32xi32, 2>, memref<64x64xi32, 1>)
              air.dma_memcpy_nd (%alloc_7[] [] [], %arg20[%arg22, %3] [%c32, %c32] [%c64_5, %c1_3]) {id = 5 : i32} : (memref<32x32xi32, 2>, memref<64x64xi32, 1>)
              air.dma_memcpy_nd (%alloc_8[] [] [], %arg21[%2, %3] [%c32, %c32] [%c64_5, %c1_3]) {id = 6 : i32} : (memref<32x32xi32, 2>, memref<64x64xi32, 1>)
              linalg.matmul {cast = #linalg.type_fn<cast_signed>, op_name = "matmul_1"} ins(%alloc_6, %alloc_7 : memref<32x32xi32, 2>, memref<32x32xi32, 2>) outs(%alloc_8 : memref<32x32xi32, 2>)
              air.dma_memcpy_nd (%arg21[%2, %3] [%c32, %c32] [%c64_5, %c1_3], %alloc_8[] [] []) {id = 7 : i32} : (memref<64x64xi32, 1>, memref<32x32xi32, 2>)
              memref.dealloc %alloc_6 : memref<32x32xi32, 2>
              memref.dealloc %alloc_7 : memref<32x32xi32, 2>
              memref.dealloc %alloc_8 : memref<32x32xi32, 2>
            }
            air.herd_terminator
          }
          air.dma_memcpy_nd (%arg13[%0, %1] [%c64, %c64] [%c512, %c1], %alloc_2[] [] []) {id = 8 : i32} : (memref<512x512xi32>, memref<64x64xi32, 1>)
          memref.dealloc %alloc_0 : memref<64x64xi32, 1>
          memref.dealloc %alloc_1 : memref<64x64xi32, 1>
          memref.dealloc %alloc_2 : memref<64x64xi32, 1>
        }
        air.segment_terminator
      }
      air.launch_terminator
    }
    return %alloc : memref<512x512xi32>
  }
}
