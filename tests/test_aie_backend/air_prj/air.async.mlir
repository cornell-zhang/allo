#map = affine_map<()[s0] -> (s0 * 64)>
#map1 = affine_map<()[s0] -> (s0 * 32)>
#set = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 1 >= 0)>
#set1 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 1 >= 0, s1 == 0)>
module {
  air.channel @channel_9 [1, 1]
  air.channel @channel_8 [2, 2]
  air.channel @channel_7 [2, 2]
  air.channel @channel_6 [1, 1]
  air.channel @channel_5 [1, 1]
  air.channel @channel_4 [1, 1]
  air.channel @channel_3 [1, 1] {broadcast_shape = [2, 1]}
  air.channel @channel_2 [1, 1] {broadcast_shape = [2, 1]}
  air.channel @channel_1 [1, 1] {broadcast_shape = [1, 2]}
  air.channel @channel_0 [1, 1] {broadcast_shape = [1, 2]}
  func.func @matmul_kernel(%arg0: memref<512x512xi32>, %arg1: memref<512x512xi32>) -> memref<512x512xi32> attributes {itypes = "ss", otypes = "s"} {
    %c8 = arith.constant 8 : index
    %c0_i32 = arith.constant 0 : i32
    %async_token, %results = air.execute -> (memref<512x512xi32>) {
      %alloc = memref.alloc() : memref<512x512xi32>
      air.execute_terminator %alloc : memref<512x512xi32>
    }
    %async_token_0 = air.execute [%async_token] {
      linalg.fill {op_name = "matmul_init_zero_0"} ins(%c0_i32 : i32) outs(%results : memref<512x512xi32>)
    }
    %0 = air.launch async [%async_token_0] (%arg2, %arg3) in (%arg4=%c8, %arg5=%c8) args(%arg6=%arg0, %arg7=%arg1, %arg8=%results) : memref<512x512xi32>, memref<512x512xi32>, memref<512x512xi32> {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %async_token_1, %results_2 = air.execute -> (index) {
        %8 = affine.apply #map()[%arg2]
        air.execute_terminator %8 : index
      }
      %async_token_3, %results_4 = air.execute -> (index) {
        %8 = affine.apply #map()[%arg3]
        air.execute_terminator %8 : index
      }
      %1 = air.channel.put async [%async_token_3, %async_token_1]  @channel_4[] (%arg8[%results_2, %results_4] [%c64, %c64] [%c512, %c1]) : (memref<512x512xi32>)
      %async_token_5, %results_6 = air.execute -> (index) {
        %8 = affine.apply #map()[%arg2]
        air.execute_terminator %8 : index
      }
      %2 = air.wait_all async 
      %3 = scf.for %arg9 = %c0 to %c512 step %c64 iter_args(%arg10 = %2) -> (!air.async.token) {
        %8 = air.channel.put async [%arg10, %async_token_5]  @channel_5[] (%arg6[%results_6, %arg9] [%c64, %c64] [%c512, %c1]) : (memref<512x512xi32>)
        scf.yield %8 : !air.async.token
      }
      %async_token_7, %results_8 = air.execute -> (index) {
        %8 = affine.apply #map()[%arg3]
        air.execute_terminator %8 : index
      }
      %4 = air.wait_all async 
      %5 = scf.for %arg9 = %c0 to %c512 step %c64 iter_args(%arg10 = %4) -> (!air.async.token) {
        %8 = air.channel.put async [%arg10, %async_token_7]  @channel_6[] (%arg7[%arg9, %results_8] [%c64, %c64] [%c512, %c1]) : (memref<512x512xi32>)
        scf.yield %8 : !air.async.token
      }
      %async_token_9, %results_10 = air.execute -> (index) {
        %8 = affine.apply #map()[%arg2]
        air.execute_terminator %8 : index
      }
      %async_token_11, %results_12 = air.execute -> (index) {
        %8 = affine.apply #map()[%arg3]
        air.execute_terminator %8 : index
      }
      %6 = air.channel.get async [%async_token_11, %async_token_9]  @channel_9[] (%arg8[%results_10, %results_12] [%c64, %c64] [%c512, %c1]) : (memref<512x512xi32>)
      %7 = air.segment async  attributes {x_loc = 0 : i64, x_size = 2 : i64, y_loc = 0 : i64, y_size = 2 : i64} {
        %c128 = arith.constant 128 : index
        %c32 = arith.constant 32 : index
        %c1_13 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c0_14 = arith.constant 0 : index
        %c512_15 = arith.constant 512 : index
        %c64_16 = arith.constant 64 : index
        %8 = air.wait_all async 
        %async_token_17, %results_18 = air.execute -> (memref<64x64xi32, 1>) {
          %alloc = memref.alloc() : memref<64x64xi32, 1>
          air.execute_terminator %alloc : memref<64x64xi32, 1>
        }
        %9 = air.channel.get async [%async_token_17, %8]  @channel_4[] (%results_18[] [] []) : (memref<64x64xi32, 1>)
        %10 = scf.for %arg9 = %c0_14 to %c512_15 step %c64_16 iter_args(%arg10 = %9) -> (!air.async.token) {
          %13 = air.herd @herd_0 async [%arg10]  tile (%arg11, %arg12) in (%arg13=%c2, %arg14=%c2) attributes {x_loc = 0 : i64, y_loc = 0 : i64} {
            %17 = air.wait_all async 
            %async_token_32, %results_33 = air.execute -> (memref<32x32xi32, 2>) {
              %alloc = memref.alloc() : memref<32x32xi32, 2>
              air.execute_terminator %alloc : memref<32x32xi32, 2>
            }
            %18 = air.channel.get async [%async_token_32, %17]  @channel_7[%arg11, %arg12] (%results_33[] [] []) : (memref<32x32xi32, 2>)
            %async_token_34, %results_35 = air.execute -> (memref<32x32xi32, 2>) {
              %alloc = memref.alloc() : memref<32x32xi32, 2>
              air.execute_terminator %alloc : memref<32x32xi32, 2>
            }
            %async_token_36, %results_37 = air.execute -> (memref<32x32xi32, 2>) {
              %alloc = memref.alloc() : memref<32x32xi32, 2>
              air.execute_terminator %alloc : memref<32x32xi32, 2>
            }
            %19 = affine.if #set()[%arg11, %arg12] -> !air.async.token {
              %24 = air.channel.get async [%async_token_34, %18]  @channel_0[%arg11, %arg12] (%results_35[] [] []) : (memref<32x32xi32, 2>)
              affine.yield %24 : !air.async.token
            } else {
              %24 = air.channel.get async [%async_token_34, %18]  @channel_1[%arg11, %arg12] (%results_35[] [] []) : (memref<32x32xi32, 2>)
              affine.yield %24 : !air.async.token
            }
            %20 = affine.if #set1()[%arg11, %arg12] -> !air.async.token {
              %24 = air.channel.get async [%async_token_36, %18]  @channel_2[%arg11, %arg12] (%results_37[] [] []) : (memref<32x32xi32, 2>)
              affine.yield %24 : !air.async.token
            } else {
              %24 = air.channel.get async [%async_token_36, %18]  @channel_3[%arg11, %arg12] (%results_37[] [] []) : (memref<32x32xi32, 2>)
              affine.yield %24 : !air.async.token
            }
            %async_token_38 = air.execute [%20, %19] {
              linalg.matmul {cast = #linalg.type_fn<cast_signed>, op_name = "matmul_1"} ins(%results_35, %results_37 : memref<32x32xi32, 2>, memref<32x32xi32, 2>) outs(%results_33 : memref<32x32xi32, 2>)
            }
            %async_token_39 = air.execute [%async_token_38] {
              memref.dealloc %results_35 : memref<32x32xi32, 2>
            }
            %async_token_40 = air.execute [%async_token_38] {
              memref.dealloc %results_37 : memref<32x32xi32, 2>
            }
            %async_token_41, %results_42 = air.execute -> (memref<32x32xi32, 2>) {
              %alloc = memref.alloc() : memref<32x32xi32, 2>
              air.execute_terminator %alloc : memref<32x32xi32, 2>
            }
            %async_token_43, %results_44 = air.execute -> (memref<32x32xi32, 2>) {
              %alloc = memref.alloc() : memref<32x32xi32, 2>
              air.execute_terminator %alloc : memref<32x32xi32, 2>
            }
            %21 = affine.if #set()[%arg11, %arg12] -> !air.async.token {
              %24 = air.channel.get async [%async_token_41, %async_token_38]  @channel_0[%arg11, %arg12] (%results_42[] [] []) : (memref<32x32xi32, 2>)
              affine.yield %24 : !air.async.token
            } else {
              %24 = air.channel.get async [%async_token_41, %async_token_38]  @channel_1[%arg11, %arg12] (%results_42[] [] []) : (memref<32x32xi32, 2>)
              affine.yield %24 : !air.async.token
            }
            %22 = affine.if #set1()[%arg11, %arg12] -> !air.async.token {
              %24 = air.channel.get async [%async_token_43, %async_token_38]  @channel_2[%arg11, %arg12] (%results_44[] [] []) : (memref<32x32xi32, 2>)
              affine.yield %24 : !air.async.token
            } else {
              %24 = air.channel.get async [%async_token_43, %async_token_38]  @channel_3[%arg11, %arg12] (%results_44[] [] []) : (memref<32x32xi32, 2>)
              affine.yield %24 : !air.async.token
            }
            %async_token_45 = air.execute [%22, %21] {
              linalg.matmul {cast = #linalg.type_fn<cast_signed>, op_name = "matmul_1"} ins(%results_42, %results_44 : memref<32x32xi32, 2>, memref<32x32xi32, 2>) outs(%results_33 : memref<32x32xi32, 2>)
            }
            %async_token_46 = air.execute [%async_token_45] {
              memref.dealloc %results_42 : memref<32x32xi32, 2>
            }
            %async_token_47 = air.execute [%async_token_45] {
              memref.dealloc %results_44 : memref<32x32xi32, 2>
            }
            %23 = air.channel.put async [%async_token_45]  @channel_8[%arg11, %arg12] (%results_33[] [] []) : (memref<32x32xi32, 2>)
            %async_token_48 = air.execute [%23] {
              memref.dealloc %results_33 : memref<32x32xi32, 2>
            }
            air.herd_terminator
          }
          %14 = scf.parallel (%arg11, %arg12) = (%c0_14, %c0_14) to (%c2, %c2) step (%c1_13, %c1_13) init (%arg10) -> !air.async.token {
            %async_token_32, %results_33 = air.execute -> (index) {
              %18 = affine.apply #map1()[%arg11]
              air.execute_terminator %18 : index
            }
            %async_token_34, %results_35 = air.execute -> (index) {
              %18 = affine.apply #map1()[%arg12]
              air.execute_terminator %18 : index
            }
            %17 = air.channel.put async [%async_token_34, %async_token_32, %arg10]  @channel_7[%arg11, %arg12] (%results_18[%results_33, %results_35] [%c32, %c32] [%c64_16, %c1_13]) : (memref<64x64xi32, 1>)
            scf.reduce(%17)  : !air.async.token {
            ^bb0(%arg13: !air.async.token, %arg14: !air.async.token):
              %18 = air.wait_all async [%arg13, %arg14] 
              scf.reduce.return %18 : !air.async.token
            }
            scf.yield
          }
          %15 = scf.parallel (%arg11, %arg12) = (%c0_14, %c0_14) to (%c2, %c2) step (%c1_13, %c1_13) init (%arg10) -> !air.async.token {
            %async_token_32, %results_33 = air.execute -> (index) {
              %18 = affine.apply #map1()[%arg11]
              air.execute_terminator %18 : index
            }
            %async_token_34, %results_35 = air.execute -> (index) {
              %18 = affine.apply #map1()[%arg12]
              air.execute_terminator %18 : index
            }
            %17 = air.channel.get async [%async_token_34, %async_token_32, %arg10]  @channel_8[%arg11, %arg12] (%results_18[%results_33, %results_35] [%c32, %c32] [%c64_16, %c1_13]) : (memref<64x64xi32, 1>)
            scf.reduce(%17)  : !air.async.token {
            ^bb0(%arg13: !air.async.token, %arg14: !air.async.token):
              %18 = air.wait_all async [%arg13, %arg14] 
              scf.reduce.return %18 : !air.async.token
            }
            scf.yield
          }
          %16 = air.wait_all async [%13, %14, %15] 
          scf.yield %16 : !air.async.token
        }
        %async_token_19, %results_20 = air.execute [%9] -> (memref<64x64xi32, 1>) {
          %alloc = memref.alloc() : memref<64x64xi32, 1>
          air.execute_terminator %alloc : memref<64x64xi32, 1>
        }
        %async_token_21, %results_22 = air.execute [%async_token_19] -> (memref<64x64xi32, 1>) {
          %alloc = memref.alloc() : memref<64x64xi32, 1>
          air.execute_terminator %alloc : memref<64x64xi32, 1>
        }
        %async_token_23, %results_24 = air.execute [%async_token_21] -> (memref<64x64xi32, 1>) {
          %alloc = memref.alloc() : memref<64x64xi32, 1>
          air.execute_terminator %alloc : memref<64x64xi32, 1>
        }
        %async_token_25, %results_26 = air.execute [%async_token_21] -> (memref<64x64xi32, 1>) {
          %alloc = memref.alloc() : memref<64x64xi32, 1>
          air.execute_terminator %alloc : memref<64x64xi32, 1>
        }
        %11:4 = scf.for %arg9 = %c0_14 to %c512_15 step %c128 iter_args(%arg10 = %async_token_23, %arg11 = %async_token_25, %arg12 = %async_token_25, %arg13 = %async_token_25) -> (!air.async.token, !air.async.token, !air.async.token, !air.async.token) {
          %13 = air.channel.get async [%arg13, %async_token_23, %arg10]  @channel_5[] (%results_24[] [] []) : (memref<64x64xi32, 1>)
          %14 = air.channel.get async [%arg13, %async_token_25, %arg10]  @channel_6[] (%results_26[] [] []) : (memref<64x64xi32, 1>)
          %15 = air.wait_all async [%arg12, %13] 
          %16 = scf.for %arg14 = %c0_14 to %c64_16 step %c32 iter_args(%arg15 = %15) -> (!air.async.token) {
            %36 = air.channel.put async [%arg15]  @channel_0[] (%results_24[%c0_14, %arg14] [%c32, %c32] [%c64_16, %c1_13]) : (memref<64x64xi32, 1>)
            scf.yield %36 : !air.async.token
          }
          %17 = air.wait_all async [%arg12, %13] 
          %18 = scf.for %arg14 = %c0_14 to %c64_16 step %c32 iter_args(%arg15 = %17) -> (!air.async.token) {
            %36 = air.channel.put async [%arg15]  @channel_1[] (%results_24[%c32, %arg14] [%c32, %c32] [%c64_16, %c1_13]) : (memref<64x64xi32, 1>)
            scf.yield %36 : !air.async.token
          }
          %19 = air.wait_all async [%arg12, %14] 
          %20 = scf.for %arg14 = %c0_14 to %c64_16 step %c32 iter_args(%arg15 = %19) -> (!air.async.token) {
            %36 = air.channel.put async [%arg15]  @channel_2[] (%results_26[%arg14, %c0_14] [%c32, %c32] [%c64_16, %c1_13]) : (memref<64x64xi32, 1>)
            scf.yield %36 : !air.async.token
          }
          %21 = air.wait_all async [%arg12, %14] 
          %22 = scf.for %arg14 = %c0_14 to %c64_16 step %c32 iter_args(%arg15 = %21) -> (!air.async.token) {
            %36 = air.channel.put async [%arg15]  @channel_3[] (%results_26[%arg14, %c32] [%c32, %c32] [%c64_16, %c1_13]) : (memref<64x64xi32, 1>)
            scf.yield %36 : !air.async.token
          }
          %23 = air.wait_all async [%13, %14, %22, %20, %18, %16] 
          %24 = air.channel.get async [%14, %13, %arg11]  @channel_5[] (%results_22[] [] []) : (memref<64x64xi32, 1>)
          %25 = air.channel.get async [%14, %13, %arg11]  @channel_6[] (%results_20[] [] []) : (memref<64x64xi32, 1>)
          %26 = air.wait_all async [%23, %24] 
          %27 = scf.for %arg14 = %c0_14 to %c64_16 step %c32 iter_args(%arg15 = %26) -> (!air.async.token) {
            %36 = air.channel.put async [%arg15]  @channel_0[] (%results_22[%c0_14, %arg14] [%c32, %c32] [%c64_16, %c1_13]) : (memref<64x64xi32, 1>)
            scf.yield %36 : !air.async.token
          }
          %28 = air.wait_all async [%23, %24] 
          %29 = scf.for %arg14 = %c0_14 to %c64_16 step %c32 iter_args(%arg15 = %28) -> (!air.async.token) {
            %36 = air.channel.put async [%arg15]  @channel_1[] (%results_22[%c32, %arg14] [%c32, %c32] [%c64_16, %c1_13]) : (memref<64x64xi32, 1>)
            scf.yield %36 : !air.async.token
          }
          %30 = air.wait_all async [%23, %25] 
          %31 = scf.for %arg14 = %c0_14 to %c64_16 step %c32 iter_args(%arg15 = %30) -> (!air.async.token) {
            %36 = air.channel.put async [%arg15]  @channel_2[] (%results_20[%arg14, %c0_14] [%c32, %c32] [%c64_16, %c1_13]) : (memref<64x64xi32, 1>)
            scf.yield %36 : !air.async.token
          }
          %32 = air.wait_all async [%23, %25] 
          %33 = scf.for %arg14 = %c0_14 to %c64_16 step %c32 iter_args(%arg15 = %32) -> (!air.async.token) {
            %36 = air.channel.put async [%arg15]  @channel_3[] (%results_20[%arg14, %c32] [%c32, %c32] [%c64_16, %c1_13]) : (memref<64x64xi32, 1>)
            scf.yield %36 : !air.async.token
          }
          %34 = air.wait_all async [%24, %25, %33, %31, %29, %27] 
          %35 = air.wait_all async [%24, %25] 
          scf.yield %23, %34, %34, %35 : !air.async.token, !air.async.token, !air.async.token, !air.async.token
        }
        %async_token_27 = air.execute [%11#1] {
          memref.dealloc %results_24 : memref<64x64xi32, 1>
        }
        %async_token_28 = air.execute [%11#1] {
          memref.dealloc %results_26 : memref<64x64xi32, 1>
        }
        %async_token_29 = air.execute [%11#1] {
          memref.dealloc %results_22 : memref<64x64xi32, 1>
        }
        %async_token_30 = air.execute [%11#1] {
          memref.dealloc %results_20 : memref<64x64xi32, 1>
        }
        %12 = air.channel.put async [%10, %11#1]  @channel_9[] (%results_18[] [] []) : (memref<64x64xi32, 1>)
        %async_token_31 = air.execute [%12] {
          memref.dealloc %results_18 : memref<64x64xi32, 1>
        }
        air.segment_terminator
      }
      air.launch_terminator
    }
    return %results : memref<512x512xi32>
  }
}
