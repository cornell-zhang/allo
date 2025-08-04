module {
  aie.device(npu1_4col) {
    func.func private @fill_zeros_bf16_64_64_vector(memref<64x64xbf16>)
    func.func private @matmul_scalar_bf16_bf16(memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>)
    func.func private @add_bf16_vector(memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>)
    func.func private @matmul_bf16_bf16(memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>)
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    %shim_noc_tile_2_0 = aie.tile(2, 0)
    %shim_noc_tile_3_0 = aie.tile(3, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    %mem_tile_1_1 = aie.tile(1, 1)
    %mem_tile_2_1 = aie.tile(2, 1)
    %mem_tile_3_1 = aie.tile(3, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_1_2 = aie.tile(1, 2)
    %tile_1_3 = aie.tile(1, 3)
    %tile_2_2 = aie.tile(2, 2)
    %tile_2_3 = aie.tile(2, 3)
    %tile_3_2 = aie.tile(3, 2)
    %tile_3_3 = aie.tile(3, 3)
    aie.objectfifo @fifo_0(%mem_tile_0_1 dimensionsToStream [<size = 16, stride = 256>, <size = 8, stride = 8>, <size = 4, stride = 64>, <size = 8, stride = 1>], {%tile_0_2, %tile_0_3}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @fifo_1(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<1x1x64x64xbf16>> 
    aie.objectfifo @fifo_2(%mem_tile_1_1 dimensionsToStream [<size = 16, stride = 256>, <size = 8, stride = 8>, <size = 4, stride = 64>, <size = 8, stride = 1>], {%tile_1_2, %tile_1_3}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @fifo_3(%shim_noc_tile_1_0, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<1x1x64x64xbf16>> 
    aie.objectfifo @fifo_4(%mem_tile_2_1 dimensionsToStream [<size = 16, stride = 256>, <size = 8, stride = 8>, <size = 4, stride = 64>, <size = 8, stride = 1>], {%tile_2_2, %tile_2_3}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @fifo_5(%shim_noc_tile_2_0, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<1x1x64x64xbf16>> 
    aie.objectfifo @fifo_6(%mem_tile_3_1 dimensionsToStream [<size = 16, stride = 256>, <size = 8, stride = 8>, <size = 4, stride = 64>, <size = 8, stride = 1>], {%tile_3_3, %tile_3_2}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @fifo_7(%shim_noc_tile_3_0, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<1x1x64x64xbf16>> 
    aie.objectfifo @fifo_8(%mem_tile_0_1 dimensionsToStream [<size = 8, stride = 512>, <size = 16, stride = 4>, <size = 8, stride = 64>, <size = 4, stride = 1>], {%tile_2_2, %tile_1_2, %tile_0_2, %tile_3_2}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @fifo_9(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<1x1x64x64xbf16>> 
    aie.objectfifo @fifo_10(%mem_tile_1_1 dimensionsToStream [<size = 8, stride = 512>, <size = 16, stride = 4>, <size = 8, stride = 64>, <size = 4, stride = 1>], {%tile_0_3, %tile_1_3, %tile_3_3, %tile_2_3}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @fifo_11(%shim_noc_tile_1_0, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<1x1x64x64xbf16>> 
    aie.objectfifo @fifo_12(%tile_0_2, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @fifo_13(%tile_0_3, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @fifo_14(%mem_tile_0_1 dimensionsToStream [<size = 16, stride = 256>, <size = 4, stride = 4>, <size = 16, stride = 16>, <size = 4, stride = 1>], {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<1x2x64x64xbf16>> 
    aie.objectfifo @fifo_15(%tile_1_2, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @fifo_16(%tile_1_3, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @fifo_17(%mem_tile_1_1 dimensionsToStream [<size = 16, stride = 256>, <size = 4, stride = 4>, <size = 16, stride = 16>, <size = 4, stride = 1>], {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<1x2x64x64xbf16>> 
    aie.objectfifo @fifo_18(%tile_2_2, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @fifo_19(%tile_2_3, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @fifo_20(%mem_tile_2_1 dimensionsToStream [<size = 16, stride = 256>, <size = 4, stride = 4>, <size = 16, stride = 16>, <size = 4, stride = 1>], {%shim_noc_tile_1_0}, 2 : i32) : !aie.objectfifo<memref<1x2x64x64xbf16>> 
    aie.objectfifo @fifo_21(%tile_3_2, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @fifo_22(%tile_3_3, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @fifo_23(%mem_tile_3_1 dimensionsToStream [<size = 16, stride = 256>, <size = 4, stride = 4>, <size = 16, stride = 16>, <size = 4, stride = 1>], {%shim_noc_tile_1_0}, 2 : i32) : !aie.objectfifo<memref<1x2x64x64xbf16>> 
    aie.objectfifo.link [@fifo_1] -> [@fifo_0]([] [])
    aie.objectfifo.link [@fifo_9] -> [@fifo_8]([] [])
    aie.objectfifo.link [@fifo_12, @fifo_13] -> [@fifo_14]([0, 4096] [])
    aie.objectfifo.link [@fifo_3] -> [@fifo_2]([] [])
    aie.objectfifo.link [@fifo_11] -> [@fifo_10]([] [])
    aie.objectfifo.link [@fifo_15, @fifo_16] -> [@fifo_17]([0, 4096] [])
    aie.objectfifo.link [@fifo_5] -> [@fifo_4]([] [])
    aie.objectfifo.link [@fifo_18, @fifo_19] -> [@fifo_20]([0, 4096] [])
    aie.objectfifo.link [@fifo_7] -> [@fifo_6]([] [])
    aie.objectfifo.link [@fifo_21, @fifo_22] -> [@fifo_23]([0, 4096] [])
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_12(Produce, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @fill_zeros_bf16_64_64_vector(%1) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
        %2 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        %4 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%3, %5, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %6 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %8 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%7, %9, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %10 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %12 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%11, %13, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %14 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %16 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%15, %17, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %18 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %20 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%19, %21, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %22 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %24 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%23, %25, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %26 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %27 = aie.objectfifo.subview.access %26[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %28 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %29 = aie.objectfifo.subview.access %28[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%27, %29, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %30 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %31 = aie.objectfifo.subview.access %30[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %32 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %33 = aie.objectfifo.subview.access %32[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%31, %33, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %34 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %35 = aie.objectfifo.subview.access %34[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %36 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %37 = aie.objectfifo.subview.access %36[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%35, %37, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %38 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %39 = aie.objectfifo.subview.access %38[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %40 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %41 = aie.objectfifo.subview.access %40[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%39, %41, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %42 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %43 = aie.objectfifo.subview.access %42[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %44 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %45 = aie.objectfifo.subview.access %44[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%43, %45, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %46 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %47 = aie.objectfifo.subview.access %46[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %48 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%47, %49, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %50 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %52 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %53 = aie.objectfifo.subview.access %52[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%51, %53, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %54 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %55 = aie.objectfifo.subview.access %54[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %56 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %57 = aie.objectfifo.subview.access %56[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%55, %57, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %58 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %59 = aie.objectfifo.subview.access %58[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %60 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %61 = aie.objectfifo.subview.access %60[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%59, %61, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %62 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %63 = aie.objectfifo.subview.access %62[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %64 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %65 = aie.objectfifo.subview.access %64[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%63, %65, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        aie.objectfifo.release @fifo_8(Consume, 1)
        aie.objectfifo.release @fifo_12(Produce, 1)
      }
      aie.end
    } {link_with = "external.o"}
    %core_0_3 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_13(Produce, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @fill_zeros_bf16_64_64_vector(%1) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
        %2 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        %4 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%3, %5, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %6 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %8 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%7, %9, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %10 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %12 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%11, %13, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %14 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %16 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%15, %17, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %18 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %20 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%19, %21, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %22 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %24 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%23, %25, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %26 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %27 = aie.objectfifo.subview.access %26[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %28 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %29 = aie.objectfifo.subview.access %28[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%27, %29, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %30 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %31 = aie.objectfifo.subview.access %30[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %32 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %33 = aie.objectfifo.subview.access %32[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%31, %33, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %34 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %35 = aie.objectfifo.subview.access %34[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %36 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %37 = aie.objectfifo.subview.access %36[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%35, %37, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %38 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %39 = aie.objectfifo.subview.access %38[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %40 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %41 = aie.objectfifo.subview.access %40[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%39, %41, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %42 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %43 = aie.objectfifo.subview.access %42[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %44 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %45 = aie.objectfifo.subview.access %44[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%43, %45, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %46 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %47 = aie.objectfifo.subview.access %46[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %48 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%47, %49, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %50 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %52 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %53 = aie.objectfifo.subview.access %52[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%51, %53, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %54 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %55 = aie.objectfifo.subview.access %54[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %56 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %57 = aie.objectfifo.subview.access %56[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%55, %57, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %58 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %59 = aie.objectfifo.subview.access %58[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %60 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %61 = aie.objectfifo.subview.access %60[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%59, %61, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %62 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %63 = aie.objectfifo.subview.access %62[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %64 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %65 = aie.objectfifo.subview.access %64[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%63, %65, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        aie.objectfifo.release @fifo_10(Consume, 1)
        aie.objectfifo.release @fifo_13(Produce, 1)
      }
      aie.end
    } {link_with = "external.o"}
    %core_1_2 = aie.core(%tile_1_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_15(Produce, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @fill_zeros_bf16_64_64_vector(%1) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
        %2 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        %4 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%3, %5, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_2(Consume, 1)
        %6 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %8 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%7, %9, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_2(Consume, 1)
        %10 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %12 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%11, %13, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_2(Consume, 1)
        %14 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %16 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%15, %17, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_2(Consume, 1)
        %18 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %20 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%19, %21, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_2(Consume, 1)
        %22 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %24 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%23, %25, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_2(Consume, 1)
        %26 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %27 = aie.objectfifo.subview.access %26[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %28 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %29 = aie.objectfifo.subview.access %28[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%27, %29, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_2(Consume, 1)
        %30 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %31 = aie.objectfifo.subview.access %30[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %32 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %33 = aie.objectfifo.subview.access %32[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%31, %33, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_2(Consume, 1)
        %34 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %35 = aie.objectfifo.subview.access %34[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %36 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %37 = aie.objectfifo.subview.access %36[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%35, %37, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_2(Consume, 1)
        %38 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %39 = aie.objectfifo.subview.access %38[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %40 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %41 = aie.objectfifo.subview.access %40[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%39, %41, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_2(Consume, 1)
        %42 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %43 = aie.objectfifo.subview.access %42[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %44 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %45 = aie.objectfifo.subview.access %44[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%43, %45, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_2(Consume, 1)
        %46 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %47 = aie.objectfifo.subview.access %46[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %48 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%47, %49, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_2(Consume, 1)
        %50 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %52 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %53 = aie.objectfifo.subview.access %52[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%51, %53, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_2(Consume, 1)
        %54 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %55 = aie.objectfifo.subview.access %54[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %56 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %57 = aie.objectfifo.subview.access %56[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%55, %57, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_2(Consume, 1)
        %58 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %59 = aie.objectfifo.subview.access %58[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %60 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %61 = aie.objectfifo.subview.access %60[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%59, %61, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_2(Consume, 1)
        %62 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %63 = aie.objectfifo.subview.access %62[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %64 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %65 = aie.objectfifo.subview.access %64[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%63, %65, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_2(Consume, 1)
        aie.objectfifo.release @fifo_8(Consume, 1)
        aie.objectfifo.release @fifo_15(Produce, 1)
      }
      aie.end
    } {link_with = "external.o"}
    %core_1_3 = aie.core(%tile_1_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_16(Produce, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @fill_zeros_bf16_64_64_vector(%1) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
        %2 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        %4 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%3, %5, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_2(Consume, 1)
        %6 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %8 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%7, %9, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_2(Consume, 1)
        %10 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %12 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%11, %13, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_2(Consume, 1)
        %14 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %16 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%15, %17, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_2(Consume, 1)
        %18 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %20 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%19, %21, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_2(Consume, 1)
        %22 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %24 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%23, %25, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_2(Consume, 1)
        %26 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %27 = aie.objectfifo.subview.access %26[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %28 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %29 = aie.objectfifo.subview.access %28[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%27, %29, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_2(Consume, 1)
        %30 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %31 = aie.objectfifo.subview.access %30[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %32 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %33 = aie.objectfifo.subview.access %32[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%31, %33, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_2(Consume, 1)
        %34 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %35 = aie.objectfifo.subview.access %34[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %36 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %37 = aie.objectfifo.subview.access %36[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%35, %37, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_2(Consume, 1)
        %38 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %39 = aie.objectfifo.subview.access %38[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %40 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %41 = aie.objectfifo.subview.access %40[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%39, %41, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_2(Consume, 1)
        %42 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %43 = aie.objectfifo.subview.access %42[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %44 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %45 = aie.objectfifo.subview.access %44[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%43, %45, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_2(Consume, 1)
        %46 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %47 = aie.objectfifo.subview.access %46[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %48 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%47, %49, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_2(Consume, 1)
        %50 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %52 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %53 = aie.objectfifo.subview.access %52[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%51, %53, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_2(Consume, 1)
        %54 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %55 = aie.objectfifo.subview.access %54[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %56 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %57 = aie.objectfifo.subview.access %56[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%55, %57, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_2(Consume, 1)
        %58 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %59 = aie.objectfifo.subview.access %58[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %60 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %61 = aie.objectfifo.subview.access %60[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%59, %61, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_2(Consume, 1)
        %62 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %63 = aie.objectfifo.subview.access %62[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %64 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %65 = aie.objectfifo.subview.access %64[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%63, %65, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_2(Consume, 1)
        aie.objectfifo.release @fifo_10(Consume, 1)
        aie.objectfifo.release @fifo_16(Produce, 1)
      }
      aie.end
    } {link_with = "external.o"}
    %core_2_2 = aie.core(%tile_2_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_18(Produce, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @fill_zeros_bf16_64_64_vector(%1) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
        %2 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        %4 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%3, %5, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_4(Consume, 1)
        %6 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %8 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%7, %9, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_4(Consume, 1)
        %10 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %12 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%11, %13, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_4(Consume, 1)
        %14 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %16 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%15, %17, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_4(Consume, 1)
        %18 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %20 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%19, %21, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_4(Consume, 1)
        %22 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %24 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%23, %25, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_4(Consume, 1)
        %26 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %27 = aie.objectfifo.subview.access %26[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %28 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %29 = aie.objectfifo.subview.access %28[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%27, %29, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_4(Consume, 1)
        %30 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %31 = aie.objectfifo.subview.access %30[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %32 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %33 = aie.objectfifo.subview.access %32[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%31, %33, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_4(Consume, 1)
        %34 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %35 = aie.objectfifo.subview.access %34[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %36 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %37 = aie.objectfifo.subview.access %36[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%35, %37, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_4(Consume, 1)
        %38 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %39 = aie.objectfifo.subview.access %38[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %40 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %41 = aie.objectfifo.subview.access %40[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%39, %41, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_4(Consume, 1)
        %42 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %43 = aie.objectfifo.subview.access %42[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %44 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %45 = aie.objectfifo.subview.access %44[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%43, %45, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_4(Consume, 1)
        %46 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %47 = aie.objectfifo.subview.access %46[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %48 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%47, %49, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_4(Consume, 1)
        %50 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %52 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %53 = aie.objectfifo.subview.access %52[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%51, %53, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_4(Consume, 1)
        %54 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %55 = aie.objectfifo.subview.access %54[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %56 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %57 = aie.objectfifo.subview.access %56[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%55, %57, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_4(Consume, 1)
        %58 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %59 = aie.objectfifo.subview.access %58[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %60 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %61 = aie.objectfifo.subview.access %60[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%59, %61, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_4(Consume, 1)
        %62 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %63 = aie.objectfifo.subview.access %62[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %64 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %65 = aie.objectfifo.subview.access %64[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%63, %65, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_4(Consume, 1)
        aie.objectfifo.release @fifo_8(Consume, 1)
        aie.objectfifo.release @fifo_18(Produce, 1)
      }
      aie.end
    } {link_with = "external.o"}
    %core_2_3 = aie.core(%tile_2_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_19(Produce, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @fill_zeros_bf16_64_64_vector(%1) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
        %2 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        %4 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%3, %5, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_4(Consume, 1)
        %6 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %8 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%7, %9, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_4(Consume, 1)
        %10 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %12 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%11, %13, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_4(Consume, 1)
        %14 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %16 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%15, %17, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_4(Consume, 1)
        %18 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %20 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%19, %21, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_4(Consume, 1)
        %22 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %24 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%23, %25, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_4(Consume, 1)
        %26 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %27 = aie.objectfifo.subview.access %26[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %28 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %29 = aie.objectfifo.subview.access %28[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%27, %29, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_4(Consume, 1)
        %30 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %31 = aie.objectfifo.subview.access %30[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %32 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %33 = aie.objectfifo.subview.access %32[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%31, %33, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_4(Consume, 1)
        %34 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %35 = aie.objectfifo.subview.access %34[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %36 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %37 = aie.objectfifo.subview.access %36[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%35, %37, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_4(Consume, 1)
        %38 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %39 = aie.objectfifo.subview.access %38[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %40 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %41 = aie.objectfifo.subview.access %40[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%39, %41, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_4(Consume, 1)
        %42 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %43 = aie.objectfifo.subview.access %42[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %44 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %45 = aie.objectfifo.subview.access %44[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%43, %45, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_4(Consume, 1)
        %46 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %47 = aie.objectfifo.subview.access %46[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %48 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%47, %49, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_4(Consume, 1)
        %50 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %52 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %53 = aie.objectfifo.subview.access %52[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%51, %53, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_4(Consume, 1)
        %54 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %55 = aie.objectfifo.subview.access %54[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %56 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %57 = aie.objectfifo.subview.access %56[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%55, %57, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_4(Consume, 1)
        %58 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %59 = aie.objectfifo.subview.access %58[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %60 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %61 = aie.objectfifo.subview.access %60[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%59, %61, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_4(Consume, 1)
        %62 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %63 = aie.objectfifo.subview.access %62[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %64 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %65 = aie.objectfifo.subview.access %64[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%63, %65, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_4(Consume, 1)
        aie.objectfifo.release @fifo_10(Consume, 1)
        aie.objectfifo.release @fifo_19(Produce, 1)
      }
      aie.end
    } {link_with = "external.o"}
    %core_3_2 = aie.core(%tile_3_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_21(Produce, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @fill_zeros_bf16_64_64_vector(%1) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
        %2 = aie.objectfifo.acquire @fifo_6(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        %4 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%3, %5, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_6(Consume, 1)
        %6 = aie.objectfifo.acquire @fifo_6(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %8 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%7, %9, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_6(Consume, 1)
        %10 = aie.objectfifo.acquire @fifo_6(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %12 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%11, %13, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_6(Consume, 1)
        %14 = aie.objectfifo.acquire @fifo_6(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %16 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%15, %17, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_6(Consume, 1)
        %18 = aie.objectfifo.acquire @fifo_6(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %20 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%19, %21, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_6(Consume, 1)
        %22 = aie.objectfifo.acquire @fifo_6(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %24 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%23, %25, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_6(Consume, 1)
        %26 = aie.objectfifo.acquire @fifo_6(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %27 = aie.objectfifo.subview.access %26[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %28 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %29 = aie.objectfifo.subview.access %28[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%27, %29, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_6(Consume, 1)
        %30 = aie.objectfifo.acquire @fifo_6(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %31 = aie.objectfifo.subview.access %30[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %32 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %33 = aie.objectfifo.subview.access %32[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%31, %33, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_6(Consume, 1)
        %34 = aie.objectfifo.acquire @fifo_6(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %35 = aie.objectfifo.subview.access %34[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %36 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %37 = aie.objectfifo.subview.access %36[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%35, %37, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_6(Consume, 1)
        %38 = aie.objectfifo.acquire @fifo_6(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %39 = aie.objectfifo.subview.access %38[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %40 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %41 = aie.objectfifo.subview.access %40[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%39, %41, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_6(Consume, 1)
        %42 = aie.objectfifo.acquire @fifo_6(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %43 = aie.objectfifo.subview.access %42[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %44 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %45 = aie.objectfifo.subview.access %44[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%43, %45, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_6(Consume, 1)
        %46 = aie.objectfifo.acquire @fifo_6(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %47 = aie.objectfifo.subview.access %46[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %48 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%47, %49, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_6(Consume, 1)
        %50 = aie.objectfifo.acquire @fifo_6(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %52 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %53 = aie.objectfifo.subview.access %52[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%51, %53, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_6(Consume, 1)
        %54 = aie.objectfifo.acquire @fifo_6(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %55 = aie.objectfifo.subview.access %54[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %56 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %57 = aie.objectfifo.subview.access %56[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%55, %57, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_6(Consume, 1)
        %58 = aie.objectfifo.acquire @fifo_6(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %59 = aie.objectfifo.subview.access %58[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %60 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %61 = aie.objectfifo.subview.access %60[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%59, %61, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_6(Consume, 1)
        %62 = aie.objectfifo.acquire @fifo_6(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %63 = aie.objectfifo.subview.access %62[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %64 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %65 = aie.objectfifo.subview.access %64[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%63, %65, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_6(Consume, 1)
        aie.objectfifo.release @fifo_8(Consume, 1)
        aie.objectfifo.release @fifo_21(Produce, 1)
      }
      aie.end
    } {link_with = "external.o"}
    %core_3_3 = aie.core(%tile_3_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_22(Produce, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @fill_zeros_bf16_64_64_vector(%1) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
        %2 = aie.objectfifo.acquire @fifo_6(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        %4 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%3, %5, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_6(Consume, 1)
        %6 = aie.objectfifo.acquire @fifo_6(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %8 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%7, %9, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_6(Consume, 1)
        %10 = aie.objectfifo.acquire @fifo_6(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %12 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%11, %13, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_6(Consume, 1)
        %14 = aie.objectfifo.acquire @fifo_6(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %16 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%15, %17, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_6(Consume, 1)
        %18 = aie.objectfifo.acquire @fifo_6(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %20 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%19, %21, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_6(Consume, 1)
        %22 = aie.objectfifo.acquire @fifo_6(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %24 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%23, %25, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_6(Consume, 1)
        %26 = aie.objectfifo.acquire @fifo_6(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %27 = aie.objectfifo.subview.access %26[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %28 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %29 = aie.objectfifo.subview.access %28[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%27, %29, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_6(Consume, 1)
        %30 = aie.objectfifo.acquire @fifo_6(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %31 = aie.objectfifo.subview.access %30[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %32 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %33 = aie.objectfifo.subview.access %32[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%31, %33, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_6(Consume, 1)
        %34 = aie.objectfifo.acquire @fifo_6(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %35 = aie.objectfifo.subview.access %34[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %36 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %37 = aie.objectfifo.subview.access %36[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%35, %37, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_6(Consume, 1)
        %38 = aie.objectfifo.acquire @fifo_6(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %39 = aie.objectfifo.subview.access %38[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %40 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %41 = aie.objectfifo.subview.access %40[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%39, %41, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_6(Consume, 1)
        %42 = aie.objectfifo.acquire @fifo_6(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %43 = aie.objectfifo.subview.access %42[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %44 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %45 = aie.objectfifo.subview.access %44[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%43, %45, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_6(Consume, 1)
        %46 = aie.objectfifo.acquire @fifo_6(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %47 = aie.objectfifo.subview.access %46[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %48 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%47, %49, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_6(Consume, 1)
        %50 = aie.objectfifo.acquire @fifo_6(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %52 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %53 = aie.objectfifo.subview.access %52[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%51, %53, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_6(Consume, 1)
        %54 = aie.objectfifo.acquire @fifo_6(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %55 = aie.objectfifo.subview.access %54[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %56 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %57 = aie.objectfifo.subview.access %56[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%55, %57, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_6(Consume, 1)
        %58 = aie.objectfifo.acquire @fifo_6(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %59 = aie.objectfifo.subview.access %58[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %60 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %61 = aie.objectfifo.subview.access %60[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%59, %61, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_6(Consume, 1)
        %62 = aie.objectfifo.acquire @fifo_6(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %63 = aie.objectfifo.subview.access %62[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %64 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %65 = aie.objectfifo.subview.access %64[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%63, %65, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_6(Consume, 1)
        aie.objectfifo.release @fifo_10(Consume, 1)
        aie.objectfifo.release @fifo_22(Produce, 1)
      }
      aie.end
    } {link_with = "external.o"}
    aiex.runtime_sequence(%arg0: memref<2048x1024xbf16>, %arg1: memref<1024x2048xbf16>, %arg2: memref<2048x2048xbf16>) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][16, 16, 64, 64][0, 64, 1024, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_1} : memref<2048x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 262144][16, 16, 64, 64][0, 64, 1024, 1]) {id = 5 : i64, issue_token = true, metadata = @fifo_1} : memref<2048x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 524288][16, 16, 64, 64][0, 64, 1024, 1]) {id = 9 : i64, issue_token = true, metadata = @fifo_1} : memref<2048x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 65536][16, 16, 64, 64][0, 64, 1024, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_3} : memref<2048x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 327680][16, 16, 64, 64][0, 64, 1024, 1]) {id = 5 : i64, issue_token = true, metadata = @fifo_3} : memref<2048x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 589824][16, 16, 64, 64][0, 64, 1024, 1]) {id = 9 : i64, issue_token = true, metadata = @fifo_3} : memref<2048x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 131072][16, 16, 64, 64][0, 64, 1024, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 393216][16, 16, 64, 64][0, 64, 1024, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 655360][16, 16, 64, 64][0, 64, 1024, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 196608][16, 16, 64, 64][0, 64, 1024, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_7} : memref<2048x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 458752][16, 16, 64, 64][0, 64, 1024, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_7} : memref<2048x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 720896][16, 16, 64, 64][0, 64, 1024, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_7} : memref<2048x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][16, 16, 64, 64][128, 131072, 2048, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_9} : memref<1024x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][16, 16, 64, 64][128, 131072, 2048, 1]) {id = 6 : i64, issue_token = true, metadata = @fifo_9} : memref<1024x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][16, 16, 64, 64][128, 131072, 2048, 1]) {id = 10 : i64, issue_token = true, metadata = @fifo_9} : memref<1024x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 64][16, 16, 64, 64][128, 131072, 2048, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_11} : memref<1024x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 64][16, 16, 64, 64][128, 131072, 2048, 1]) {id = 6 : i64, issue_token = true, metadata = @fifo_11} : memref<1024x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 64][16, 16, 64, 64][128, 131072, 2048, 1]) {id = 10 : i64, issue_token = true, metadata = @fifo_11} : memref<1024x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 0][16, 2, 64, 64][128, 64, 2048, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_14} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 524288][16, 2, 64, 64][128, 64, 2048, 1]) {id = 7 : i64, issue_token = true, metadata = @fifo_14} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 1048576][16, 2, 64, 64][128, 64, 2048, 1]) {id = 11 : i64, issue_token = true, metadata = @fifo_14} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 131072][16, 2, 64, 64][128, 64, 2048, 1]) {id = 4 : i64, issue_token = true, metadata = @fifo_17} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 655360][16, 2, 64, 64][128, 64, 2048, 1]) {id = 8 : i64, issue_token = true, metadata = @fifo_17} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 1179648][16, 2, 64, 64][128, 64, 2048, 1]) {id = 12 : i64, issue_token = true, metadata = @fifo_17} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 262144][16, 2, 64, 64][128, 64, 2048, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_20} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 786432][16, 2, 64, 64][128, 64, 2048, 1]) {id = 7 : i64, issue_token = true, metadata = @fifo_20} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 1310720][16, 2, 64, 64][128, 64, 2048, 1]) {id = 11 : i64, issue_token = true, metadata = @fifo_20} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 393216][16, 2, 64, 64][128, 64, 2048, 1]) {id = 4 : i64, issue_token = true, metadata = @fifo_23} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 917504][16, 2, 64, 64][128, 64, 2048, 1]) {id = 8 : i64, issue_token = true, metadata = @fifo_23} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 1441792][16, 2, 64, 64][128, 64, 2048, 1]) {id = 12 : i64, issue_token = true, metadata = @fifo_23} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_14}
      aiex.npu.dma_wait {symbol = @fifo_14}
      aiex.npu.dma_wait {symbol = @fifo_14}
      aiex.npu.dma_wait {symbol = @fifo_17}
      aiex.npu.dma_wait {symbol = @fifo_17}
      aiex.npu.dma_wait {symbol = @fifo_17}
      aiex.npu.dma_wait {symbol = @fifo_20}
      aiex.npu.dma_wait {symbol = @fifo_20}
      aiex.npu.dma_wait {symbol = @fifo_20}
      aiex.npu.dma_wait {symbol = @fifo_23}
      aiex.npu.dma_wait {symbol = @fifo_23}
      aiex.npu.dma_wait {symbol = @fifo_23}
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 786432][16, 16, 64, 64][0, 64, 1024, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_1} : memref<2048x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 1048576][16, 16, 64, 64][0, 64, 1024, 1]) {id = 5 : i64, issue_token = true, metadata = @fifo_1} : memref<2048x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 1310720][16, 16, 64, 64][0, 64, 1024, 1]) {id = 9 : i64, issue_token = true, metadata = @fifo_1} : memref<2048x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 851968][16, 16, 64, 64][0, 64, 1024, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_3} : memref<2048x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 1114112][16, 16, 64, 64][0, 64, 1024, 1]) {id = 5 : i64, issue_token = true, metadata = @fifo_3} : memref<2048x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 1376256][16, 16, 64, 64][0, 64, 1024, 1]) {id = 9 : i64, issue_token = true, metadata = @fifo_3} : memref<2048x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 917504][16, 16, 64, 64][0, 64, 1024, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 1179648][16, 16, 64, 64][0, 64, 1024, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 1441792][16, 16, 64, 64][0, 64, 1024, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 983040][16, 16, 64, 64][0, 64, 1024, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_7} : memref<2048x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 1245184][16, 16, 64, 64][0, 64, 1024, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_7} : memref<2048x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 1507328][16, 16, 64, 64][0, 64, 1024, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_7} : memref<2048x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][16, 16, 64, 64][128, 131072, 2048, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_9} : memref<1024x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][16, 16, 64, 64][128, 131072, 2048, 1]) {id = 6 : i64, issue_token = true, metadata = @fifo_9} : memref<1024x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][16, 16, 64, 64][128, 131072, 2048, 1]) {id = 10 : i64, issue_token = true, metadata = @fifo_9} : memref<1024x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 64][16, 16, 64, 64][128, 131072, 2048, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_11} : memref<1024x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 64][16, 16, 64, 64][128, 131072, 2048, 1]) {id = 6 : i64, issue_token = true, metadata = @fifo_11} : memref<1024x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 64][16, 16, 64, 64][128, 131072, 2048, 1]) {id = 10 : i64, issue_token = true, metadata = @fifo_11} : memref<1024x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 1572864][16, 2, 64, 64][128, 64, 2048, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_14} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 2097152][16, 2, 64, 64][128, 64, 2048, 1]) {id = 7 : i64, issue_token = true, metadata = @fifo_14} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 2621440][16, 2, 64, 64][128, 64, 2048, 1]) {id = 11 : i64, issue_token = true, metadata = @fifo_14} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 1703936][16, 2, 64, 64][128, 64, 2048, 1]) {id = 4 : i64, issue_token = true, metadata = @fifo_17} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 2228224][16, 2, 64, 64][128, 64, 2048, 1]) {id = 8 : i64, issue_token = true, metadata = @fifo_17} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 2752512][16, 2, 64, 64][128, 64, 2048, 1]) {id = 12 : i64, issue_token = true, metadata = @fifo_17} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 1835008][16, 2, 64, 64][128, 64, 2048, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_20} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 2359296][16, 2, 64, 64][128, 64, 2048, 1]) {id = 7 : i64, issue_token = true, metadata = @fifo_20} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 2883584][16, 2, 64, 64][128, 64, 2048, 1]) {id = 11 : i64, issue_token = true, metadata = @fifo_20} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 1966080][16, 2, 64, 64][128, 64, 2048, 1]) {id = 4 : i64, issue_token = true, metadata = @fifo_23} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 2490368][16, 2, 64, 64][128, 64, 2048, 1]) {id = 8 : i64, issue_token = true, metadata = @fifo_23} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 3014656][16, 2, 64, 64][128, 64, 2048, 1]) {id = 12 : i64, issue_token = true, metadata = @fifo_23} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_14}
      aiex.npu.dma_wait {symbol = @fifo_14}
      aiex.npu.dma_wait {symbol = @fifo_14}
      aiex.npu.dma_wait {symbol = @fifo_17}
      aiex.npu.dma_wait {symbol = @fifo_17}
      aiex.npu.dma_wait {symbol = @fifo_17}
      aiex.npu.dma_wait {symbol = @fifo_20}
      aiex.npu.dma_wait {symbol = @fifo_20}
      aiex.npu.dma_wait {symbol = @fifo_20}
      aiex.npu.dma_wait {symbol = @fifo_23}
      aiex.npu.dma_wait {symbol = @fifo_23}
      aiex.npu.dma_wait {symbol = @fifo_23}
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 1572864][16, 16, 64, 64][0, 64, 1024, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_1} : memref<2048x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 1835008][16, 16, 64, 64][0, 64, 1024, 1]) {id = 5 : i64, issue_token = true, metadata = @fifo_1} : memref<2048x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 1638400][16, 16, 64, 64][0, 64, 1024, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_3} : memref<2048x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 1900544][16, 16, 64, 64][0, 64, 1024, 1]) {id = 5 : i64, issue_token = true, metadata = @fifo_3} : memref<2048x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 1703936][16, 16, 64, 64][0, 64, 1024, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 1966080][16, 16, 64, 64][0, 64, 1024, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 1769472][16, 16, 64, 64][0, 64, 1024, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_7} : memref<2048x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 2031616][16, 16, 64, 64][0, 64, 1024, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_7} : memref<2048x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][16, 16, 64, 64][128, 131072, 2048, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_9} : memref<1024x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][16, 16, 64, 64][128, 131072, 2048, 1]) {id = 6 : i64, issue_token = true, metadata = @fifo_9} : memref<1024x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 64][16, 16, 64, 64][128, 131072, 2048, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_11} : memref<1024x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 64][16, 16, 64, 64][128, 131072, 2048, 1]) {id = 6 : i64, issue_token = true, metadata = @fifo_11} : memref<1024x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 3145728][16, 2, 64, 64][128, 64, 2048, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_14} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 3670016][16, 2, 64, 64][128, 64, 2048, 1]) {id = 7 : i64, issue_token = true, metadata = @fifo_14} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 3276800][16, 2, 64, 64][128, 64, 2048, 1]) {id = 4 : i64, issue_token = true, metadata = @fifo_17} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 3801088][16, 2, 64, 64][128, 64, 2048, 1]) {id = 8 : i64, issue_token = true, metadata = @fifo_17} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 3407872][16, 2, 64, 64][128, 64, 2048, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_20} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 3932160][16, 2, 64, 64][128, 64, 2048, 1]) {id = 7 : i64, issue_token = true, metadata = @fifo_20} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 3538944][16, 2, 64, 64][128, 64, 2048, 1]) {id = 4 : i64, issue_token = true, metadata = @fifo_23} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 4063232][16, 2, 64, 64][128, 64, 2048, 1]) {id = 8 : i64, issue_token = true, metadata = @fifo_23} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_14}
      aiex.npu.dma_wait {symbol = @fifo_14}
      aiex.npu.dma_wait {symbol = @fifo_17}
      aiex.npu.dma_wait {symbol = @fifo_17}
      aiex.npu.dma_wait {symbol = @fifo_20}
      aiex.npu.dma_wait {symbol = @fifo_20}
      aiex.npu.dma_wait {symbol = @fifo_23}
      aiex.npu.dma_wait {symbol = @fifo_23}
      aie.end
    }
  }
}
