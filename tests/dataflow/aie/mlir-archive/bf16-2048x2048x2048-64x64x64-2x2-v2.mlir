module {
  aie.device(npu1_1col) {
    func.func private @fill_zeros_bf16_64_64_vector(memref<64x64xbf16>)
    func.func private @matmul_scalar_bf16_bf16(memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>)
    func.func private @add_bf16_vector(memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>)
    func.func private @matmul_bf16_bf16(memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>)
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_4 = aie.tile(0, 4)
    %tile_0_5 = aie.tile(0, 5)
    aie.objectfifo @fifo_0(%mem_tile_0_1 dimensionsToStream [<size = 16, stride = 256>, <size = 8, stride = 8>, <size = 4, stride = 64>, <size = 8, stride = 1>], {%tile_0_2, %tile_0_3}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @fifo_1(%mem_tile_0_1 dimensionsToStream [<size = 16, stride = 256>, <size = 8, stride = 8>, <size = 4, stride = 64>, <size = 8, stride = 1>], {%tile_0_4, %tile_0_5}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @fifo_2(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<2x1x64x64xbf16>> 
    aie.objectfifo @fifo_3(%mem_tile_0_1 dimensionsToStream [<size = 8, stride = 512>, <size = 16, stride = 4>, <size = 8, stride = 64>, <size = 4, stride = 1>], {%tile_0_4, %tile_0_2}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @fifo_4(%mem_tile_0_1 dimensionsToStream [<size = 8, stride = 512>, <size = 16, stride = 4>, <size = 8, stride = 64>, <size = 4, stride = 1>], {%tile_0_5, %tile_0_3}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @fifo_5(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<1x2x64x64xbf16>> 
    aie.objectfifo @fifo_6(%tile_0_2, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @fifo_7(%tile_0_3, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @fifo_8(%mem_tile_0_1 dimensionsToStream [<size = 16, stride = 256>, <size = 4, stride = 4>, <size = 16, stride = 16>, <size = 4, stride = 1>], {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<1x2x64x64xbf16>> 
    aie.objectfifo @fifo_9(%tile_0_4, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @fifo_10(%tile_0_5, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @fifo_11(%mem_tile_0_1 dimensionsToStream [<size = 16, stride = 256>, <size = 4, stride = 4>, <size = 16, stride = 16>, <size = 4, stride = 1>], {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<1x2x64x64xbf16>> 
    aie.objectfifo.link [@fifo_2] -> [@fifo_0, @fifo_1]([] [0, 4096])
    aie.objectfifo.link [@fifo_5] -> [@fifo_3, @fifo_4]([] [0, 4096])
    aie.objectfifo.link [@fifo_6, @fifo_7] -> [@fifo_8]([0, 4096] [])
    aie.objectfifo.link [@fifo_9, @fifo_10] -> [@fifo_11]([0, 4096] [])
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_6(Produce, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @fill_zeros_bf16_64_64_vector(%1) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
        %2 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        %4 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%3, %5, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %6 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_3(Consume, 1)
        %8 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%7, %9, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %10 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_3(Consume, 1)
        %12 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%11, %13, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %14 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_3(Consume, 1)
        %16 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%15, %17, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %18 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_3(Consume, 1)
        %20 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%19, %21, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %22 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_3(Consume, 1)
        %24 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%23, %25, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %26 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %27 = aie.objectfifo.subview.access %26[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_3(Consume, 1)
        %28 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %29 = aie.objectfifo.subview.access %28[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%27, %29, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %30 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %31 = aie.objectfifo.subview.access %30[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_3(Consume, 1)
        %32 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %33 = aie.objectfifo.subview.access %32[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%31, %33, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %34 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %35 = aie.objectfifo.subview.access %34[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_3(Consume, 1)
        %36 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %37 = aie.objectfifo.subview.access %36[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%35, %37, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %38 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %39 = aie.objectfifo.subview.access %38[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_3(Consume, 1)
        %40 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %41 = aie.objectfifo.subview.access %40[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%39, %41, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %42 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %43 = aie.objectfifo.subview.access %42[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_3(Consume, 1)
        %44 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %45 = aie.objectfifo.subview.access %44[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%43, %45, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %46 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %47 = aie.objectfifo.subview.access %46[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_3(Consume, 1)
        %48 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%47, %49, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %50 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_3(Consume, 1)
        %52 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %53 = aie.objectfifo.subview.access %52[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%51, %53, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %54 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %55 = aie.objectfifo.subview.access %54[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_3(Consume, 1)
        %56 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %57 = aie.objectfifo.subview.access %56[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%55, %57, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %58 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %59 = aie.objectfifo.subview.access %58[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_3(Consume, 1)
        %60 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %61 = aie.objectfifo.subview.access %60[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%59, %61, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %62 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %63 = aie.objectfifo.subview.access %62[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_3(Consume, 1)
        %64 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %65 = aie.objectfifo.subview.access %64[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%63, %65, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %66 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %67 = aie.objectfifo.subview.access %66[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_3(Consume, 1)
        %68 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %69 = aie.objectfifo.subview.access %68[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%67, %69, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %70 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %71 = aie.objectfifo.subview.access %70[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_3(Consume, 1)
        %72 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %73 = aie.objectfifo.subview.access %72[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%71, %73, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %74 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %75 = aie.objectfifo.subview.access %74[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_3(Consume, 1)
        %76 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %77 = aie.objectfifo.subview.access %76[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%75, %77, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %78 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %79 = aie.objectfifo.subview.access %78[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_3(Consume, 1)
        %80 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %81 = aie.objectfifo.subview.access %80[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%79, %81, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %82 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %83 = aie.objectfifo.subview.access %82[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_3(Consume, 1)
        %84 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %85 = aie.objectfifo.subview.access %84[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%83, %85, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %86 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %87 = aie.objectfifo.subview.access %86[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_3(Consume, 1)
        %88 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %89 = aie.objectfifo.subview.access %88[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%87, %89, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %90 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %91 = aie.objectfifo.subview.access %90[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_3(Consume, 1)
        %92 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %93 = aie.objectfifo.subview.access %92[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%91, %93, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %94 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %95 = aie.objectfifo.subview.access %94[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_3(Consume, 1)
        %96 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %97 = aie.objectfifo.subview.access %96[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%95, %97, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %98 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %99 = aie.objectfifo.subview.access %98[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_3(Consume, 1)
        %100 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %101 = aie.objectfifo.subview.access %100[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%99, %101, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %102 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %103 = aie.objectfifo.subview.access %102[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_3(Consume, 1)
        %104 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %105 = aie.objectfifo.subview.access %104[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%103, %105, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %106 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %107 = aie.objectfifo.subview.access %106[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_3(Consume, 1)
        %108 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %109 = aie.objectfifo.subview.access %108[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%107, %109, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %110 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %111 = aie.objectfifo.subview.access %110[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_3(Consume, 1)
        %112 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %113 = aie.objectfifo.subview.access %112[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%111, %113, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %114 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %115 = aie.objectfifo.subview.access %114[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_3(Consume, 1)
        %116 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %117 = aie.objectfifo.subview.access %116[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%115, %117, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %118 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %119 = aie.objectfifo.subview.access %118[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_3(Consume, 1)
        %120 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %121 = aie.objectfifo.subview.access %120[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%119, %121, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %122 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %123 = aie.objectfifo.subview.access %122[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_3(Consume, 1)
        %124 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %125 = aie.objectfifo.subview.access %124[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%123, %125, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %126 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %127 = aie.objectfifo.subview.access %126[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_3(Consume, 1)
        %128 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %129 = aie.objectfifo.subview.access %128[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%127, %129, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        aie.objectfifo.release @fifo_3(Consume, 1)
        aie.objectfifo.release @fifo_6(Produce, 1)
      }
      aie.end
    } {link_with = "external.o"}
    %core_0_3 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_7(Produce, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @fill_zeros_bf16_64_64_vector(%1) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
        %2 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        %4 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%3, %5, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %6 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_4(Consume, 1)
        %8 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%7, %9, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %10 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_4(Consume, 1)
        %12 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%11, %13, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %14 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_4(Consume, 1)
        %16 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%15, %17, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %18 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_4(Consume, 1)
        %20 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%19, %21, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %22 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_4(Consume, 1)
        %24 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%23, %25, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %26 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %27 = aie.objectfifo.subview.access %26[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_4(Consume, 1)
        %28 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %29 = aie.objectfifo.subview.access %28[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%27, %29, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %30 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %31 = aie.objectfifo.subview.access %30[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_4(Consume, 1)
        %32 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %33 = aie.objectfifo.subview.access %32[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%31, %33, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %34 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %35 = aie.objectfifo.subview.access %34[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_4(Consume, 1)
        %36 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %37 = aie.objectfifo.subview.access %36[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%35, %37, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %38 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %39 = aie.objectfifo.subview.access %38[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_4(Consume, 1)
        %40 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %41 = aie.objectfifo.subview.access %40[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%39, %41, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %42 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %43 = aie.objectfifo.subview.access %42[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_4(Consume, 1)
        %44 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %45 = aie.objectfifo.subview.access %44[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%43, %45, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %46 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %47 = aie.objectfifo.subview.access %46[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_4(Consume, 1)
        %48 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%47, %49, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %50 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_4(Consume, 1)
        %52 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %53 = aie.objectfifo.subview.access %52[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%51, %53, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %54 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %55 = aie.objectfifo.subview.access %54[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_4(Consume, 1)
        %56 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %57 = aie.objectfifo.subview.access %56[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%55, %57, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %58 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %59 = aie.objectfifo.subview.access %58[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_4(Consume, 1)
        %60 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %61 = aie.objectfifo.subview.access %60[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%59, %61, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %62 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %63 = aie.objectfifo.subview.access %62[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_4(Consume, 1)
        %64 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %65 = aie.objectfifo.subview.access %64[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%63, %65, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %66 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %67 = aie.objectfifo.subview.access %66[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_4(Consume, 1)
        %68 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %69 = aie.objectfifo.subview.access %68[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%67, %69, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %70 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %71 = aie.objectfifo.subview.access %70[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_4(Consume, 1)
        %72 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %73 = aie.objectfifo.subview.access %72[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%71, %73, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %74 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %75 = aie.objectfifo.subview.access %74[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_4(Consume, 1)
        %76 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %77 = aie.objectfifo.subview.access %76[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%75, %77, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %78 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %79 = aie.objectfifo.subview.access %78[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_4(Consume, 1)
        %80 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %81 = aie.objectfifo.subview.access %80[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%79, %81, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %82 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %83 = aie.objectfifo.subview.access %82[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_4(Consume, 1)
        %84 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %85 = aie.objectfifo.subview.access %84[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%83, %85, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %86 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %87 = aie.objectfifo.subview.access %86[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_4(Consume, 1)
        %88 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %89 = aie.objectfifo.subview.access %88[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%87, %89, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %90 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %91 = aie.objectfifo.subview.access %90[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_4(Consume, 1)
        %92 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %93 = aie.objectfifo.subview.access %92[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%91, %93, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %94 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %95 = aie.objectfifo.subview.access %94[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_4(Consume, 1)
        %96 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %97 = aie.objectfifo.subview.access %96[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%95, %97, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %98 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %99 = aie.objectfifo.subview.access %98[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_4(Consume, 1)
        %100 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %101 = aie.objectfifo.subview.access %100[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%99, %101, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %102 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %103 = aie.objectfifo.subview.access %102[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_4(Consume, 1)
        %104 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %105 = aie.objectfifo.subview.access %104[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%103, %105, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %106 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %107 = aie.objectfifo.subview.access %106[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_4(Consume, 1)
        %108 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %109 = aie.objectfifo.subview.access %108[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%107, %109, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %110 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %111 = aie.objectfifo.subview.access %110[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_4(Consume, 1)
        %112 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %113 = aie.objectfifo.subview.access %112[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%111, %113, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %114 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %115 = aie.objectfifo.subview.access %114[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_4(Consume, 1)
        %116 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %117 = aie.objectfifo.subview.access %116[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%115, %117, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %118 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %119 = aie.objectfifo.subview.access %118[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_4(Consume, 1)
        %120 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %121 = aie.objectfifo.subview.access %120[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%119, %121, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %122 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %123 = aie.objectfifo.subview.access %122[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_4(Consume, 1)
        %124 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %125 = aie.objectfifo.subview.access %124[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%123, %125, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %126 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %127 = aie.objectfifo.subview.access %126[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_4(Consume, 1)
        %128 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %129 = aie.objectfifo.subview.access %128[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%127, %129, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        aie.objectfifo.release @fifo_4(Consume, 1)
        aie.objectfifo.release @fifo_7(Produce, 1)
      }
      aie.end
    } {link_with = "external.o"}
    %core_0_4 = aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_9(Produce, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @fill_zeros_bf16_64_64_vector(%1) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
        %2 = aie.objectfifo.acquire @fifo_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        %4 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%3, %5, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_1(Consume, 1)
        %6 = aie.objectfifo.acquire @fifo_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_3(Consume, 1)
        %8 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%7, %9, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_1(Consume, 1)
        %10 = aie.objectfifo.acquire @fifo_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_3(Consume, 1)
        %12 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%11, %13, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_1(Consume, 1)
        %14 = aie.objectfifo.acquire @fifo_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_3(Consume, 1)
        %16 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%15, %17, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_1(Consume, 1)
        %18 = aie.objectfifo.acquire @fifo_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_3(Consume, 1)
        %20 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%19, %21, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_1(Consume, 1)
        %22 = aie.objectfifo.acquire @fifo_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_3(Consume, 1)
        %24 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%23, %25, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_1(Consume, 1)
        %26 = aie.objectfifo.acquire @fifo_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %27 = aie.objectfifo.subview.access %26[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_3(Consume, 1)
        %28 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %29 = aie.objectfifo.subview.access %28[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%27, %29, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_1(Consume, 1)
        %30 = aie.objectfifo.acquire @fifo_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %31 = aie.objectfifo.subview.access %30[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_3(Consume, 1)
        %32 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %33 = aie.objectfifo.subview.access %32[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%31, %33, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_1(Consume, 1)
        %34 = aie.objectfifo.acquire @fifo_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %35 = aie.objectfifo.subview.access %34[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_3(Consume, 1)
        %36 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %37 = aie.objectfifo.subview.access %36[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%35, %37, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_1(Consume, 1)
        %38 = aie.objectfifo.acquire @fifo_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %39 = aie.objectfifo.subview.access %38[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_3(Consume, 1)
        %40 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %41 = aie.objectfifo.subview.access %40[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%39, %41, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_1(Consume, 1)
        %42 = aie.objectfifo.acquire @fifo_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %43 = aie.objectfifo.subview.access %42[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_3(Consume, 1)
        %44 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %45 = aie.objectfifo.subview.access %44[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%43, %45, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_1(Consume, 1)
        %46 = aie.objectfifo.acquire @fifo_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %47 = aie.objectfifo.subview.access %46[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_3(Consume, 1)
        %48 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%47, %49, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_1(Consume, 1)
        %50 = aie.objectfifo.acquire @fifo_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_3(Consume, 1)
        %52 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %53 = aie.objectfifo.subview.access %52[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%51, %53, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_1(Consume, 1)
        %54 = aie.objectfifo.acquire @fifo_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %55 = aie.objectfifo.subview.access %54[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_3(Consume, 1)
        %56 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %57 = aie.objectfifo.subview.access %56[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%55, %57, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_1(Consume, 1)
        %58 = aie.objectfifo.acquire @fifo_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %59 = aie.objectfifo.subview.access %58[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_3(Consume, 1)
        %60 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %61 = aie.objectfifo.subview.access %60[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%59, %61, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_1(Consume, 1)
        %62 = aie.objectfifo.acquire @fifo_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %63 = aie.objectfifo.subview.access %62[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_3(Consume, 1)
        %64 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %65 = aie.objectfifo.subview.access %64[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%63, %65, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_1(Consume, 1)
        %66 = aie.objectfifo.acquire @fifo_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %67 = aie.objectfifo.subview.access %66[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_3(Consume, 1)
        %68 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %69 = aie.objectfifo.subview.access %68[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%67, %69, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_1(Consume, 1)
        %70 = aie.objectfifo.acquire @fifo_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %71 = aie.objectfifo.subview.access %70[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_3(Consume, 1)
        %72 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %73 = aie.objectfifo.subview.access %72[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%71, %73, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_1(Consume, 1)
        %74 = aie.objectfifo.acquire @fifo_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %75 = aie.objectfifo.subview.access %74[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_3(Consume, 1)
        %76 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %77 = aie.objectfifo.subview.access %76[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%75, %77, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_1(Consume, 1)
        %78 = aie.objectfifo.acquire @fifo_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %79 = aie.objectfifo.subview.access %78[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_3(Consume, 1)
        %80 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %81 = aie.objectfifo.subview.access %80[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%79, %81, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_1(Consume, 1)
        %82 = aie.objectfifo.acquire @fifo_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %83 = aie.objectfifo.subview.access %82[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_3(Consume, 1)
        %84 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %85 = aie.objectfifo.subview.access %84[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%83, %85, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_1(Consume, 1)
        %86 = aie.objectfifo.acquire @fifo_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %87 = aie.objectfifo.subview.access %86[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_3(Consume, 1)
        %88 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %89 = aie.objectfifo.subview.access %88[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%87, %89, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_1(Consume, 1)
        %90 = aie.objectfifo.acquire @fifo_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %91 = aie.objectfifo.subview.access %90[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_3(Consume, 1)
        %92 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %93 = aie.objectfifo.subview.access %92[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%91, %93, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_1(Consume, 1)
        %94 = aie.objectfifo.acquire @fifo_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %95 = aie.objectfifo.subview.access %94[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_3(Consume, 1)
        %96 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %97 = aie.objectfifo.subview.access %96[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%95, %97, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_1(Consume, 1)
        %98 = aie.objectfifo.acquire @fifo_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %99 = aie.objectfifo.subview.access %98[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_3(Consume, 1)
        %100 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %101 = aie.objectfifo.subview.access %100[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%99, %101, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_1(Consume, 1)
        %102 = aie.objectfifo.acquire @fifo_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %103 = aie.objectfifo.subview.access %102[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_3(Consume, 1)
        %104 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %105 = aie.objectfifo.subview.access %104[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%103, %105, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_1(Consume, 1)
        %106 = aie.objectfifo.acquire @fifo_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %107 = aie.objectfifo.subview.access %106[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_3(Consume, 1)
        %108 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %109 = aie.objectfifo.subview.access %108[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%107, %109, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_1(Consume, 1)
        %110 = aie.objectfifo.acquire @fifo_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %111 = aie.objectfifo.subview.access %110[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_3(Consume, 1)
        %112 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %113 = aie.objectfifo.subview.access %112[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%111, %113, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_1(Consume, 1)
        %114 = aie.objectfifo.acquire @fifo_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %115 = aie.objectfifo.subview.access %114[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_3(Consume, 1)
        %116 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %117 = aie.objectfifo.subview.access %116[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%115, %117, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_1(Consume, 1)
        %118 = aie.objectfifo.acquire @fifo_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %119 = aie.objectfifo.subview.access %118[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_3(Consume, 1)
        %120 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %121 = aie.objectfifo.subview.access %120[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%119, %121, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_1(Consume, 1)
        %122 = aie.objectfifo.acquire @fifo_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %123 = aie.objectfifo.subview.access %122[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_3(Consume, 1)
        %124 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %125 = aie.objectfifo.subview.access %124[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%123, %125, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_1(Consume, 1)
        %126 = aie.objectfifo.acquire @fifo_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %127 = aie.objectfifo.subview.access %126[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_3(Consume, 1)
        %128 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %129 = aie.objectfifo.subview.access %128[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%127, %129, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_1(Consume, 1)
        aie.objectfifo.release @fifo_3(Consume, 1)
        aie.objectfifo.release @fifo_9(Produce, 1)
      }
      aie.end
    } {link_with = "external.o"}
    %core_0_5 = aie.core(%tile_0_5) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_10(Produce, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @fill_zeros_bf16_64_64_vector(%1) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
        %2 = aie.objectfifo.acquire @fifo_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        %4 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%3, %5, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_1(Consume, 1)
        %6 = aie.objectfifo.acquire @fifo_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_4(Consume, 1)
        %8 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%7, %9, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_1(Consume, 1)
        %10 = aie.objectfifo.acquire @fifo_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_4(Consume, 1)
        %12 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%11, %13, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_1(Consume, 1)
        %14 = aie.objectfifo.acquire @fifo_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_4(Consume, 1)
        %16 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%15, %17, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_1(Consume, 1)
        %18 = aie.objectfifo.acquire @fifo_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_4(Consume, 1)
        %20 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%19, %21, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_1(Consume, 1)
        %22 = aie.objectfifo.acquire @fifo_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_4(Consume, 1)
        %24 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%23, %25, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_1(Consume, 1)
        %26 = aie.objectfifo.acquire @fifo_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %27 = aie.objectfifo.subview.access %26[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_4(Consume, 1)
        %28 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %29 = aie.objectfifo.subview.access %28[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%27, %29, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_1(Consume, 1)
        %30 = aie.objectfifo.acquire @fifo_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %31 = aie.objectfifo.subview.access %30[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_4(Consume, 1)
        %32 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %33 = aie.objectfifo.subview.access %32[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%31, %33, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_1(Consume, 1)
        %34 = aie.objectfifo.acquire @fifo_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %35 = aie.objectfifo.subview.access %34[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_4(Consume, 1)
        %36 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %37 = aie.objectfifo.subview.access %36[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%35, %37, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_1(Consume, 1)
        %38 = aie.objectfifo.acquire @fifo_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %39 = aie.objectfifo.subview.access %38[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_4(Consume, 1)
        %40 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %41 = aie.objectfifo.subview.access %40[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%39, %41, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_1(Consume, 1)
        %42 = aie.objectfifo.acquire @fifo_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %43 = aie.objectfifo.subview.access %42[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_4(Consume, 1)
        %44 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %45 = aie.objectfifo.subview.access %44[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%43, %45, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_1(Consume, 1)
        %46 = aie.objectfifo.acquire @fifo_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %47 = aie.objectfifo.subview.access %46[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_4(Consume, 1)
        %48 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%47, %49, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_1(Consume, 1)
        %50 = aie.objectfifo.acquire @fifo_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %51 = aie.objectfifo.subview.access %50[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_4(Consume, 1)
        %52 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %53 = aie.objectfifo.subview.access %52[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%51, %53, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_1(Consume, 1)
        %54 = aie.objectfifo.acquire @fifo_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %55 = aie.objectfifo.subview.access %54[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_4(Consume, 1)
        %56 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %57 = aie.objectfifo.subview.access %56[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%55, %57, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_1(Consume, 1)
        %58 = aie.objectfifo.acquire @fifo_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %59 = aie.objectfifo.subview.access %58[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_4(Consume, 1)
        %60 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %61 = aie.objectfifo.subview.access %60[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%59, %61, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_1(Consume, 1)
        %62 = aie.objectfifo.acquire @fifo_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %63 = aie.objectfifo.subview.access %62[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_4(Consume, 1)
        %64 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %65 = aie.objectfifo.subview.access %64[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%63, %65, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_1(Consume, 1)
        %66 = aie.objectfifo.acquire @fifo_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %67 = aie.objectfifo.subview.access %66[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_4(Consume, 1)
        %68 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %69 = aie.objectfifo.subview.access %68[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%67, %69, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_1(Consume, 1)
        %70 = aie.objectfifo.acquire @fifo_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %71 = aie.objectfifo.subview.access %70[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_4(Consume, 1)
        %72 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %73 = aie.objectfifo.subview.access %72[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%71, %73, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_1(Consume, 1)
        %74 = aie.objectfifo.acquire @fifo_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %75 = aie.objectfifo.subview.access %74[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_4(Consume, 1)
        %76 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %77 = aie.objectfifo.subview.access %76[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%75, %77, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_1(Consume, 1)
        %78 = aie.objectfifo.acquire @fifo_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %79 = aie.objectfifo.subview.access %78[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_4(Consume, 1)
        %80 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %81 = aie.objectfifo.subview.access %80[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%79, %81, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_1(Consume, 1)
        %82 = aie.objectfifo.acquire @fifo_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %83 = aie.objectfifo.subview.access %82[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_4(Consume, 1)
        %84 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %85 = aie.objectfifo.subview.access %84[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%83, %85, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_1(Consume, 1)
        %86 = aie.objectfifo.acquire @fifo_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %87 = aie.objectfifo.subview.access %86[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_4(Consume, 1)
        %88 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %89 = aie.objectfifo.subview.access %88[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%87, %89, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_1(Consume, 1)
        %90 = aie.objectfifo.acquire @fifo_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %91 = aie.objectfifo.subview.access %90[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_4(Consume, 1)
        %92 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %93 = aie.objectfifo.subview.access %92[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%91, %93, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_1(Consume, 1)
        %94 = aie.objectfifo.acquire @fifo_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %95 = aie.objectfifo.subview.access %94[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_4(Consume, 1)
        %96 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %97 = aie.objectfifo.subview.access %96[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%95, %97, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_1(Consume, 1)
        %98 = aie.objectfifo.acquire @fifo_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %99 = aie.objectfifo.subview.access %98[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_4(Consume, 1)
        %100 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %101 = aie.objectfifo.subview.access %100[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%99, %101, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_1(Consume, 1)
        %102 = aie.objectfifo.acquire @fifo_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %103 = aie.objectfifo.subview.access %102[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_4(Consume, 1)
        %104 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %105 = aie.objectfifo.subview.access %104[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%103, %105, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_1(Consume, 1)
        %106 = aie.objectfifo.acquire @fifo_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %107 = aie.objectfifo.subview.access %106[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_4(Consume, 1)
        %108 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %109 = aie.objectfifo.subview.access %108[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%107, %109, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_1(Consume, 1)
        %110 = aie.objectfifo.acquire @fifo_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %111 = aie.objectfifo.subview.access %110[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_4(Consume, 1)
        %112 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %113 = aie.objectfifo.subview.access %112[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%111, %113, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_1(Consume, 1)
        %114 = aie.objectfifo.acquire @fifo_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %115 = aie.objectfifo.subview.access %114[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_4(Consume, 1)
        %116 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %117 = aie.objectfifo.subview.access %116[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%115, %117, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_1(Consume, 1)
        %118 = aie.objectfifo.acquire @fifo_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %119 = aie.objectfifo.subview.access %118[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_4(Consume, 1)
        %120 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %121 = aie.objectfifo.subview.access %120[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%119, %121, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_1(Consume, 1)
        %122 = aie.objectfifo.acquire @fifo_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %123 = aie.objectfifo.subview.access %122[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_4(Consume, 1)
        %124 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %125 = aie.objectfifo.subview.access %124[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%123, %125, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_1(Consume, 1)
        %126 = aie.objectfifo.acquire @fifo_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %127 = aie.objectfifo.subview.access %126[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        aie.objectfifo.release @fifo_4(Consume, 1)
        %128 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %129 = aie.objectfifo.subview.access %128[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%127, %129, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_1(Consume, 1)
        aie.objectfifo.release @fifo_4(Consume, 1)
        aie.objectfifo.release @fifo_10(Produce, 1)
      }
      aie.end
    } {link_with = "external.o"}
    aiex.runtime_sequence(%arg0: memref<2048x2048xbf16>, %arg1: memref<2048x2048xbf16>, %arg2: memref<2048x2048xbf16>) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 4 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 8 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 12 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 2, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 5 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 4, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 9 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 6, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 13 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 2, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 6 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 4, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 10 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 6, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 14 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[1, 0, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[1, 2, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 7 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[1, 4, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 11 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[1, 6, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 15 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 8, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 8, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[1, 8, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 7 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 10, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 4 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 10, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 5 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[1, 10, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 6 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 11 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 12, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 10 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 12, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 9 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[1, 12, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 8 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 14 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 14, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 15 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 14, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 13 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[1, 14, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 12 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 16, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 16, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[1, 16, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 4 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 18, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 7 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 18, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 5 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[1, 18, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 6 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 10 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 20, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 11 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 20, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 8 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[1, 20, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 9 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 15 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 22, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 14 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 22, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 12 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[1, 22, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 13 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 24, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 24, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[1, 24, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 7 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 26, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 4 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 26, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 5 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[1, 26, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 6 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 11 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 28, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 10 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 28, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 9 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[1, 28, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 8 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 14 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 30, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 15 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 30, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 13 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[1, 30, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 12 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 2, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[2, 0, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[3, 0, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 2, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 4 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 2, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 7 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[2, 2, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 5 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[3, 2, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 6 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 2, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 10 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 4, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 11 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[2, 4, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 8 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[3, 4, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 9 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 2, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 15 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 6, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 14 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[2, 6, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 12 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[3, 6, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 13 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 2, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 8, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[2, 8, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[3, 8, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 2, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 7 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 10, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 4 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[2, 10, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 5 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[3, 10, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 6 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 2, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 11 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 12, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 10 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[2, 12, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 9 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[3, 12, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 8 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 2, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 14 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 14, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 15 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[2, 14, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 13 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[3, 14, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 12 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 2, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 16, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[2, 16, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[3, 16, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 2, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 4 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 18, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 7 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[2, 18, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 5 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[3, 18, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 6 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 2, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 10 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 20, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 11 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[2, 20, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 8 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[3, 20, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 9 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 2, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 15 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 22, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 14 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[2, 22, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 12 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[3, 22, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 13 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 2, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 24, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[2, 24, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[3, 24, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 2, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 7 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 26, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 4 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[2, 26, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 5 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[3, 26, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 6 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 2, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 11 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 28, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 10 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[2, 28, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 9 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[3, 28, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 8 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 2, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 14 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 30, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 15 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[2, 30, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 13 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[3, 30, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 12 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 4, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[4, 0, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[5, 0, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 4, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 4 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 2, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 7 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[4, 2, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 5 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[5, 2, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 6 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 4, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 10 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 4, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 11 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[4, 4, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 8 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[5, 4, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 9 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 4, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 15 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 6, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 14 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[4, 6, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 12 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[5, 6, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 13 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 4, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 8, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[4, 8, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[5, 8, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 4, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 7 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 10, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 4 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[4, 10, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 5 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[5, 10, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 6 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 4, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 11 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 12, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 10 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[4, 12, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 9 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[5, 12, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 8 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 4, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 14 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 14, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 15 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[4, 14, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 13 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[5, 14, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 12 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 4, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 16, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[4, 16, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[5, 16, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 4, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 4 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 18, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 7 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[4, 18, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 5 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[5, 18, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 6 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 4, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 10 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 20, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 11 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[4, 20, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 8 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[5, 20, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 9 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 4, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 15 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 22, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 14 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[4, 22, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 12 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[5, 22, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 13 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 4, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 24, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[4, 24, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[5, 24, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 4, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 7 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 26, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 4 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[4, 26, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 5 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[5, 26, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 6 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 4, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 11 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 28, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 10 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[4, 28, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 9 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[5, 28, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 8 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 4, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 14 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 30, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 15 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[4, 30, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 13 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[5, 30, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 12 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 6, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[6, 0, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[7, 0, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 6, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 4 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 2, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 7 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[6, 2, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 5 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[7, 2, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 6 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 6, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 10 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 4, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 11 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[6, 4, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 8 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[7, 4, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 9 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 6, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 15 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 6, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 14 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[6, 6, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 12 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[7, 6, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 13 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 6, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 8, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[6, 8, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[7, 8, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 6, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 7 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 10, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 4 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[6, 10, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 5 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[7, 10, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 6 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 6, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 11 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 12, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 10 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[6, 12, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 9 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[7, 12, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 8 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 6, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 14 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 14, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 15 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[6, 14, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 13 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[7, 14, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 12 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 6, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 16, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[6, 16, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[7, 16, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 6, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 4 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 18, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 7 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[6, 18, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 5 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[7, 18, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 6 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 6, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 10 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 20, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 11 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[6, 20, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 8 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[7, 20, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 9 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 6, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 15 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 22, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 14 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[6, 22, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 12 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[7, 22, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 13 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 6, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 24, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[6, 24, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[7, 24, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 6, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 7 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 26, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 4 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[6, 26, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 5 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[7, 26, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 6 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 6, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 11 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 28, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 10 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[6, 28, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 9 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[7, 28, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 8 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 6, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 14 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 30, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 15 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[6, 30, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 13 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[7, 30, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 12 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 8, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[8, 0, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[9, 0, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 8, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 4 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 2, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 7 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[8, 2, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 5 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[9, 2, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 6 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 8, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 10 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 4, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 11 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[8, 4, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 8 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[9, 4, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 9 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 8, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 15 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 6, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 14 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[8, 6, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 12 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[9, 6, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 13 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 8, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 8, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[8, 8, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[9, 8, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 8, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 7 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 10, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 4 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[8, 10, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 5 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[9, 10, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 6 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 8, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 11 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 12, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 10 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[8, 12, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 9 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[9, 12, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 8 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 8, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 14 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 14, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 15 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[8, 14, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 13 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[9, 14, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 12 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 8, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 16, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[8, 16, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[9, 16, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 8, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 4 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 18, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 7 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[8, 18, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 5 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[9, 18, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 6 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 8, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 10 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 20, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 11 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[8, 20, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 8 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[9, 20, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 9 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 8, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 15 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 22, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 14 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[8, 22, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 12 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[9, 22, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 13 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 8, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 24, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[8, 24, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[9, 24, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 8, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 7 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 26, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 4 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[8, 26, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 5 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[9, 26, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 6 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 8, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 11 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 28, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 10 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[8, 28, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 9 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[9, 28, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 8 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 8, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 14 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 30, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 15 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[8, 30, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 13 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[9, 30, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 12 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 10, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[10, 0, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[11, 0, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 10, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 4 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 2, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 7 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[10, 2, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 5 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[11, 2, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 6 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 10, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 10 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 4, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 11 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[10, 4, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 8 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[11, 4, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 9 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 10, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 15 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 6, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 14 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[10, 6, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 12 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[11, 6, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 13 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 10, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 8, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[10, 8, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[11, 8, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 10, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 7 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 10, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 4 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[10, 10, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 5 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[11, 10, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 6 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 10, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 11 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 12, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 10 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[10, 12, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 9 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[11, 12, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 8 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 10, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 14 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 14, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 15 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[10, 14, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 13 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[11, 14, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 12 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 10, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 16, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[10, 16, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[11, 16, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 10, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 4 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 18, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 7 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[10, 18, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 5 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[11, 18, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 6 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 10, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 10 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 20, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 11 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[10, 20, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 8 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[11, 20, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 9 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 10, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 15 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 22, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 14 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[10, 22, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 12 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[11, 22, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 13 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 10, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 24, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[10, 24, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[11, 24, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 10, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 7 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 26, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 4 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[10, 26, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 5 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[11, 26, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 6 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 10, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 11 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 28, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 10 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[10, 28, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 9 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[11, 28, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 8 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 10, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 14 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 30, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 15 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[10, 30, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 13 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[11, 30, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 12 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 12, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[12, 0, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[13, 0, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 12, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 4 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 2, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 7 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[12, 2, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 5 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[13, 2, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 6 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 12, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 10 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 4, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 11 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[12, 4, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 8 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[13, 4, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 9 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 12, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 15 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 6, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 14 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[12, 6, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 12 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[13, 6, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 13 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 12, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 8, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[12, 8, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[13, 8, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 12, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 7 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 10, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 4 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[12, 10, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 5 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[13, 10, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 6 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 12, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 11 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 12, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 10 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[12, 12, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 9 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[13, 12, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 8 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 12, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 14 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 14, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 15 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[12, 14, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 13 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[13, 14, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 12 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 12, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 16, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[12, 16, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[13, 16, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 12, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 4 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 18, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 7 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[12, 18, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 5 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[13, 18, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 6 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 12, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 10 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 20, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 11 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[12, 20, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 8 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[13, 20, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 9 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 12, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 15 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 22, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 14 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[12, 22, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 12 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[13, 22, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 13 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 12, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 24, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[12, 24, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[13, 24, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 12, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 7 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 26, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 4 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[12, 26, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 5 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[13, 26, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 6 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 12, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 11 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 28, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 10 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[12, 28, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 9 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[13, 28, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 8 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 12, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 14 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 30, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 15 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[12, 30, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 13 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[13, 30, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 12 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 14, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[14, 0, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[15, 0, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 14, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 4 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 2, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 7 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[14, 2, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 5 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[15, 2, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 6 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 14, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 10 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 4, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 11 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[14, 4, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 8 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[15, 4, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 9 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 14, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 15 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 6, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 14 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[14, 6, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 12 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[15, 6, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 13 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 14, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 8, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[14, 8, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[15, 8, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 14, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 7 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 10, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 4 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[14, 10, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 5 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[15, 10, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 6 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 14, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 11 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 12, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 10 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[14, 12, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 9 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[15, 12, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 8 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 14, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 14 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 14, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 15 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[14, 14, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 13 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[15, 14, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 12 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 14, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 16, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[14, 16, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[15, 16, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 14, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 4 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 18, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 7 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[14, 18, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 5 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[15, 18, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 6 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 14, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 10 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 20, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 11 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[14, 20, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 8 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[15, 20, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 9 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 14, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 15 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 22, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 14 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[14, 22, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 12 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[15, 22, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 13 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 14, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 24, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[14, 24, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[15, 24, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 14, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 7 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 26, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 4 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[14, 26, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 5 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[15, 26, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 6 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 14, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 11 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 28, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 10 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[14, 28, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 9 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[15, 28, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 8 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 14, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 14 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 30, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 15 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[14, 30, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 13 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[15, 30, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 12 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 16, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[16, 0, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[17, 0, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 16, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 4 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 2, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 7 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[16, 2, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 5 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[17, 2, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 6 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 16, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 10 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 4, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 11 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[16, 4, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 8 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[17, 4, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 9 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 16, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 15 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 6, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 14 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[16, 6, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 12 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[17, 6, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 13 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 16, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 8, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[16, 8, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[17, 8, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 16, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 7 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 10, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 4 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[16, 10, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 5 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[17, 10, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 6 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 16, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 11 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 12, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 10 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[16, 12, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 9 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[17, 12, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 8 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 16, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 14 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 14, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 15 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[16, 14, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 13 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[17, 14, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 12 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 16, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 16, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[16, 16, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[17, 16, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 16, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 4 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 18, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 7 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[16, 18, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 5 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[17, 18, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 6 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 16, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 10 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 20, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 11 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[16, 20, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 8 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[17, 20, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 9 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 16, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 15 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 22, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 14 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[16, 22, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 12 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[17, 22, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 13 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 16, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 24, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[16, 24, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[17, 24, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 16, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 7 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 26, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 4 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[16, 26, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 5 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[17, 26, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 6 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 16, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 11 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 28, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 10 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[16, 28, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 9 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[17, 28, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 8 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 16, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 14 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 30, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 15 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[16, 30, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 13 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[17, 30, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 12 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 18, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[18, 0, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[19, 0, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 18, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 4 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 2, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 7 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[18, 2, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 5 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[19, 2, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 6 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 18, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 10 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 4, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 11 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[18, 4, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 8 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[19, 4, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 9 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 18, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 15 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 6, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 14 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[18, 6, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 12 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[19, 6, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 13 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 18, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 8, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[18, 8, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[19, 8, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 18, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 7 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 10, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 4 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[18, 10, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 5 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[19, 10, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 6 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 18, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 11 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 12, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 10 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[18, 12, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 9 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[19, 12, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 8 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 18, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 14 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 14, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 15 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[18, 14, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 13 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[19, 14, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 12 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 18, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 16, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[18, 16, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[19, 16, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 18, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 4 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 18, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 7 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[18, 18, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 5 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[19, 18, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 6 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 18, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 10 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 20, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 11 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[18, 20, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 8 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[19, 20, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 9 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 18, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 15 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 22, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 14 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[18, 22, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 12 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[19, 22, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 13 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 18, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 24, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[18, 24, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[19, 24, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 18, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 7 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 26, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 4 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[18, 26, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 5 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[19, 26, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 6 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 18, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 11 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 28, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 10 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[18, 28, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 9 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[19, 28, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 8 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 18, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 14 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 30, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 15 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[18, 30, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 13 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[19, 30, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 12 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 20, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[20, 0, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[21, 0, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 20, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 4 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 2, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 7 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[20, 2, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 5 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[21, 2, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 6 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 20, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 10 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 4, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 11 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[20, 4, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 8 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[21, 4, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 9 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 20, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 15 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 6, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 14 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[20, 6, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 12 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[21, 6, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 13 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 20, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 8, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[20, 8, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[21, 8, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 20, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 7 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 10, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 4 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[20, 10, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 5 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[21, 10, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 6 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 20, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 11 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 12, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 10 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[20, 12, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 9 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[21, 12, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 8 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 20, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 14 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 14, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 15 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[20, 14, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 13 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[21, 14, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 12 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 20, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 16, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[20, 16, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[21, 16, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 20, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 4 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 18, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 7 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[20, 18, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 5 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[21, 18, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 6 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 20, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 10 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 20, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 11 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[20, 20, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 8 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[21, 20, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 9 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 20, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 15 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 22, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 14 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[20, 22, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 12 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[21, 22, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 13 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 20, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 24, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[20, 24, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[21, 24, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 20, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 7 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 26, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 4 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[20, 26, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 5 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[21, 26, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 6 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 20, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 11 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 28, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 10 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[20, 28, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 9 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[21, 28, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 8 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 20, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 14 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 30, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 15 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[20, 30, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 13 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[21, 30, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 12 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 22, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[22, 0, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[23, 0, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 22, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 4 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 2, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 7 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[22, 2, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 5 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[23, 2, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 6 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 22, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 10 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 4, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 11 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[22, 4, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 8 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[23, 4, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 9 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 22, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 15 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 6, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 14 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[22, 6, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 12 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[23, 6, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 13 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 22, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 8, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[22, 8, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[23, 8, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 22, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 7 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 10, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 4 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[22, 10, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 5 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[23, 10, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 6 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 22, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 11 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 12, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 10 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[22, 12, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 9 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[23, 12, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 8 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 22, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 14 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 14, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 15 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[22, 14, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 13 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[23, 14, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 12 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 22, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 16, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[22, 16, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[23, 16, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 22, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 4 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 18, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 7 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[22, 18, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 5 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[23, 18, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 6 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 22, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 10 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 20, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 11 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[22, 20, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 8 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[23, 20, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 9 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 22, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 15 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 22, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 14 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[22, 22, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 12 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[23, 22, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 13 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 22, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 24, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[22, 24, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[23, 24, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 22, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 7 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 26, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 4 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[22, 26, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 5 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[23, 26, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 6 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 22, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 11 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 28, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 10 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[22, 28, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 9 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[23, 28, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 8 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 22, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 14 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 30, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 15 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[22, 30, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 13 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[23, 30, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 12 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 24, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[24, 0, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[25, 0, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 24, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 4 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 2, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 7 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[24, 2, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 5 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[25, 2, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 6 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 24, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 10 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 4, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 11 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[24, 4, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 8 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[25, 4, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 9 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 24, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 15 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 6, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 14 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[24, 6, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 12 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[25, 6, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 13 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 24, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 8, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[24, 8, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[25, 8, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 24, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 7 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 10, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 4 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[24, 10, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 5 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[25, 10, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 6 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 24, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 11 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 12, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 10 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[24, 12, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 9 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[25, 12, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 8 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 24, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 14 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 14, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 15 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[24, 14, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 13 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[25, 14, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 12 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 24, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 16, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[24, 16, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[25, 16, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 24, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 4 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 18, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 7 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[24, 18, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 5 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[25, 18, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 6 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 24, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 10 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 20, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 11 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[24, 20, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 8 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[25, 20, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 9 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 24, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 15 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 22, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 14 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[24, 22, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 12 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[25, 22, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 13 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 24, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 24, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[24, 24, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[25, 24, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 24, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 7 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 26, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 4 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[24, 26, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 5 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[25, 26, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 6 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 24, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 11 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 28, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 10 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[24, 28, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 9 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[25, 28, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 8 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 24, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 14 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 30, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 15 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[24, 30, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 13 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[25, 30, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 12 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 26, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[26, 0, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[27, 0, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 26, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 4 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 2, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 7 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[26, 2, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 5 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[27, 2, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 6 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 26, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 10 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 4, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 11 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[26, 4, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 8 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[27, 4, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 9 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 26, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 15 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 6, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 14 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[26, 6, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 12 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[27, 6, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 13 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 26, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 8, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[26, 8, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[27, 8, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 26, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 7 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 10, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 4 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[26, 10, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 5 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[27, 10, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 6 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 26, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 11 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 12, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 10 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[26, 12, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 9 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[27, 12, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 8 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 26, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 14 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 14, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 15 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[26, 14, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 13 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[27, 14, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 12 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 26, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 16, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[26, 16, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[27, 16, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 26, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 4 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 18, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 7 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[26, 18, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 5 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[27, 18, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 6 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 26, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 10 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 20, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 11 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[26, 20, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 8 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[27, 20, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 9 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 26, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 15 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 22, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 14 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[26, 22, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 12 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[27, 22, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 13 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 26, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 24, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[26, 24, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[27, 24, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 26, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 7 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 26, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 4 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[26, 26, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 5 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[27, 26, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 6 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 26, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 11 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 28, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 10 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[26, 28, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 9 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[27, 28, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 8 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 26, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 14 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 30, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 15 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[26, 30, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 13 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[27, 30, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 12 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 28, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[28, 0, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[29, 0, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 28, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 4 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 2, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 7 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[28, 2, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 5 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[29, 2, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 6 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 28, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 10 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 4, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 11 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[28, 4, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 8 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[29, 4, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 9 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 28, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 15 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 6, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 14 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[28, 6, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 12 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[29, 6, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 13 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 28, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 8, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[28, 8, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[29, 8, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 28, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 7 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 10, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 4 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[28, 10, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 5 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[29, 10, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 6 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 28, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 11 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 12, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 10 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[28, 12, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 9 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[29, 12, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 8 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 28, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 14 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 14, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 15 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[28, 14, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 13 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[29, 14, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 12 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 28, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 16, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[28, 16, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[29, 16, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 28, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 4 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 18, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 7 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[28, 18, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 5 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[29, 18, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 6 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 28, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 10 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 20, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 11 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[28, 20, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 8 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[29, 20, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 9 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 28, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 15 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 22, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 14 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[28, 22, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 12 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[29, 22, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 13 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 28, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 24, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[28, 24, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[29, 24, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 28, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 7 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 26, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 4 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[28, 26, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 5 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[29, 26, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 6 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 28, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 11 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 28, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 10 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[28, 28, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 9 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[29, 28, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 8 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 28, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 14 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 30, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 15 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[28, 30, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 13 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[29, 30, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 12 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 30, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[30, 0, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[31, 0, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 30, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 4 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 2, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 7 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[30, 2, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 5 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[31, 2, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 6 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 30, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 10 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 4, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 11 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[30, 4, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 8 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[31, 4, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 9 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 30, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 15 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 6, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 14 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[30, 6, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 12 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[31, 6, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 13 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 30, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 8, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[30, 8, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[31, 8, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 30, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 7 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 10, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 4 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[30, 10, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 5 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[31, 10, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 6 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 30, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 11 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 12, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 10 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[30, 12, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 9 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[31, 12, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 8 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 30, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 14 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 14, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 15 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[30, 14, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 13 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[31, 14, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 12 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 30, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 16, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[30, 16, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[31, 16, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 30, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 4 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 18, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 7 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[30, 18, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 5 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[31, 18, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 6 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 30, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 10 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 20, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 11 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[30, 20, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 8 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[31, 20, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 9 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 30, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 15 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 22, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 14 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[30, 22, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 12 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[31, 22, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 13 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 30, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 24, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[30, 24, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[31, 24, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 30, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 7 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 26, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 4 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[30, 26, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 5 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[31, 26, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 6 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 30, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 11 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 28, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 10 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[30, 28, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 9 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[31, 28, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 8 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_memcpy_nd(%arg0[0, 30, 0, 0][32, 2, 64, 64][64, 131072, 2048, 1]) {id = 14 : i64, issue_token = true, metadata = @fifo_2} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 30, 0, 0][32, 2, 64, 64][131072, 64, 2048, 1]) {id = 15 : i64, issue_token = true, metadata = @fifo_5} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[30, 30, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 13 : i64, issue_token = true, metadata = @fifo_8} : memref<2048x2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[31, 30, 0, 0][1, 2, 64, 64][131072, 64, 2048, 1]) {id = 12 : i64, issue_token = true, metadata = @fifo_11} : memref<2048x2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_wait {symbol = @fifo_8}
      aiex.npu.dma_wait {symbol = @fifo_11}
      aie.end
    }
  }
}
