#map = affine_map<(d0) -> (0)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<() -> (0)>
#map3 = affine_map<() -> (4)>
#map4 = affine_map<(d0, d1) -> (d0, d1)>
#map5 = affine_map<(d0, d1, d2) -> (d0, 0, d1)>
#map6 = affine_map<(d0, d1) -> (d1, d0)>
#map7 = affine_map<(d0, d1, d2) -> (d0, 4, d2)>
"builtin.module"() ({
  "func.func"() <{function_type = (memref<4xi8>, memref<4xi8>, memref<4xi8>, memref<4xi8>, memref<4x4xi16>, index, index) -> (), sym_name = "PE_kernel", sym_visibility = "private"}> ({
  ^bb0(%arg11: memref<4xi8>, %arg12: memref<4xi8>, %arg13: memref<4xi8>, %arg14: memref<4xi8>, %arg15: memref<4x4xi16>, %arg16: index, %arg17: index):
    %18 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1xi16>
    %19 = "arith.constant"() <{value = 0 : i16}> : () -> i16
    %20 = "arith.constant"() <{value = 0 : index}> : () -> index
    "affine.store"(%19, %18, %20) <{map = #map}> : (i16, memref<1xi16>, index) -> ()
    "affine.for"() <{lowerBoundMap = #map2, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map3}> ({
    ^bb0(%arg18: index):
      %22 = "affine.load"(%arg11, %arg18) <{map = #map1}> : (memref<4xi8>, index) -> i8
      %23 = "affine.load"(%arg12, %arg18) <{map = #map1}> : (memref<4xi8>, index) -> i8
      %24 = "arith.extsi"(%22) : (i8) -> i16
      %25 = "arith.extsi"(%23) : (i8) -> i16
      %26 = "arith.muli"(%24, %25) <{overflowFlags = #arith.overflow<none>}> : (i16, i16) -> i16
      %27 = "affine.load"(%18, %20) <{map = #map1}> : (memref<1xi16>, index) -> i16
      %28 = "arith.addi"(%27, %26) <{overflowFlags = #arith.overflow<none>}> : (i16, i16) -> i16
      "affine.store"(%28, %18, %20) <{map = #map1}> : (i16, memref<1xi16>, index) -> ()
      "affine.store"(%22, %arg13, %arg18) <{map = #map1}> : (i8, memref<4xi8>, index) -> ()
      "affine.store"(%23, %arg14, %arg18) <{map = #map1}> : (i8, memref<4xi8>, index) -> ()
      "affine.yield"() : () -> ()
    }) {loop_name = "k"} : () -> ()
    %21 = "affine.load"(%18, %20) <{map = #map1}> : (memref<1xi16>, index) -> i16
    "affine.store"(%21, %arg15, %arg16, %arg17) <{map = #map4}> : (i16, memref<4x4xi16>, index, index) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = (memref<4x4xi8>, memref<4x4xi8>, memref<4x4xi16>) -> (), sym_name = "gemm"}> ({
  ^bb0(%arg0: memref<4x4xi8>, %arg1: memref<4x4xi8>, %arg2: memref<4x4xi16>):
    %0 = "arith.constant"() <{value = 4 : index}> : () -> index
    %1 = "arith.constant"() <{value = 4 : index}> : () -> index
    %2 = "arith.constant"() <{value = 0 : index}> : () -> index
    %3 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<4x5x4xi8>
    %4 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<4x5x4xi8>
    %5 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<4xi8>
    %6 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<4xi8>
    "affine.for"() <{lowerBoundMap = #map2, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map3}> ({
    ^bb0(%arg8: index):
      "affine.for"() <{lowerBoundMap = #map2, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map3}> ({
      ^bb0(%arg10: index):
        %17 = "affine.load"(%arg0, %arg10, %arg8) <{map = #map4}> : (memref<4x4xi8>, index, index) -> i8
        "affine.store"(%17, %3, %arg10, %2, %arg8) <{map = #map5}> : (i8, memref<4x5x4xi8>, index, index, index) -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      "affine.for"() <{lowerBoundMap = #map2, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map3}> ({
      ^bb0(%arg9: index):
        %16 = "affine.load"(%arg1, %arg8, %arg9) <{map = #map6}> : (memref<4x4xi8>, index, index) -> i8
        "affine.store"(%16, %4, %arg9, %2, %arg8) <{map = #map5}> : (i8, memref<4x5x4xi8>, index, index, index) -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      "affine.yield"() : () -> ()
    }) {name = "data_load"} : () -> ()
    "affine.for"() <{lowerBoundMap = #map2, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map3}> ({
    ^bb0(%arg6: index):
      "affine.for"() <{lowerBoundMap = #map2, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map3}> ({
      ^bb0(%arg7: index):
        %9 = "arith.constant"() <{value = 1 : index}> : () -> index
        %10 = "arith.addi"(%arg6, %9) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %11 = "arith.addi"(%arg7, %9) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %12 = "memref.subview"(%3, %arg6, %arg7) <{operandSegmentSizes = array<i32: 1, 2, 0, 0>, static_offsets = array<i64: -9223372036854775808, -9223372036854775808, 0>, static_sizes = array<i64: 1, 1, 4>, static_strides = array<i64: 1, 1, 1>}> : (memref<4x5x4xi8>, index, index) -> memref<4xi8, strided<[1], offset: ?>>
        %13 = "memref.subview"(%3, %arg6, %11) <{operandSegmentSizes = array<i32: 1, 2, 0, 0>, static_offsets = array<i64: -9223372036854775808, -9223372036854775808, 0>, static_sizes = array<i64: 1, 1, 4>, static_strides = array<i64: 1, 1, 1>}> : (memref<4x5x4xi8>, index, index) -> memref<4xi8, strided<[1], offset: ?>>
        %14 = "memref.subview"(%4, %arg7, %arg6) <{operandSegmentSizes = array<i32: 1, 2, 0, 0>, static_offsets = array<i64: -9223372036854775808, -9223372036854775808, 0>, static_sizes = array<i64: 1, 1, 4>, static_strides = array<i64: 1, 1, 1>}> : (memref<4x5x4xi8>, index, index) -> memref<4xi8, strided<[1], offset: ?>>
        %15 = "memref.subview"(%4, %arg7, %10) <{operandSegmentSizes = array<i32: 1, 2, 0, 0>, static_offsets = array<i64: -9223372036854775808, -9223372036854775808, 0>, static_sizes = array<i64: 1, 1, 4>, static_strides = array<i64: 1, 1, 1>}> : (memref<4x5x4xi8>, index, index) -> memref<4xi8, strided<[1], offset: ?>>
        "func.call"(%12, %14, %13, %15, %arg2, %arg6, %arg7) <{callee = @PE_kernel}> : (memref<4xi8, strided<[1], offset: ?>>, memref<4xi8, strided<[1], offset: ?>>, memref<4xi8, strided<[1], offset: ?>>, memref<4xi8, strided<[1], offset: ?>>, memref<4x4xi16>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {loop_name = "j"} : () -> ()
      "affine.yield"() : () -> ()
    }) {loop_name = "i", op_name = "PE"} : () -> ()
    "affine.for"() <{lowerBoundMap = #map2, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map3}> ({
    ^bb0(%arg3: index):
      "affine.for"() <{lowerBoundMap = #map2, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map3}> ({
      ^bb0(%arg5: index):
        %8 = "affine.load"(%3, %arg5, %1, %arg3) <{map = #map7}> : (memref<4x5x4xi8>, index, index, index) -> i8
        "affine.store"(%8, %5, %arg5) <{map = #map1}> : (i8, memref<4xi8>, index) -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      "affine.for"() <{lowerBoundMap = #map2, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map3}> ({
      ^bb0(%arg4: index):
        %7 = "affine.load"(%4, %arg4, %0, %arg3) <{map = #map7}> : (memref<4x5x4xi8>, index, index, index) -> i8
        "affine.store"(%7, %6, %arg4) <{map = #map1}> : (i8, memref<4xi8>, index) -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      "affine.yield"() : () -> ()
    }) {name = "data_drain"} : () -> ()
    "func.return"() : () -> ()
  }) {itypes = "sss", otypes = ""} : () -> ()
}) : () -> ()