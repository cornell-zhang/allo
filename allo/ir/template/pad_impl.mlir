// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
// https://github.com/llvm/llvm-project/blob/main/mlir/test/Dialect/Linalg/generalize-pad-tensor.mlir

func.func @pad(%input: memref<1x3x224x224xf32>) -> memref<1x3x228x228xf32> {
    %alloc = memref.alloc() : memref<1x3x228x228xf32>
    %c0 = arith.constant 0.0 : f32

    linalg.generic {
        indexing_maps = [
            affine_map<(i, j, k, l) -> (i, j, k + 2, l + 2)>,
            affine_map<(i, j, k, l) -> (i, j, k, l)>
        ],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    }
    ins(%input : memref<1x3x224x224xf32>) outs(%alloc : memref<1x3x228x228xf32>) {
        ^bb(%i: index, %j: index, %k: index, %l: index, %in: f32, %out: f32):
            %c2 = arith.constant 2 : index
            %c225 = arith.constant 225 : index

            %k_ge_2 = arith.cmpi sge, %k, %c2 : index
            %k_le_225 = arith.cmpi sle, %k, %c225 : index
            %l_ge_2 = arith.cmpi sge, %l, %c2 : index
            %l_le_225 = arith.cmpi sle, %l, %c225 : index

            %k_in_range = arith.and %k_ge_2, %k_le_225 : index
            %l_in_range = arith.and %l_ge_2, %l_le_225 : index

            %is_in_range = arith.and %k_in_range, %l_in_range : index

            %value = arith.select %is_in_range, %in, %c0 : f32
            linalg.yield %value : f32
    }
    return %alloc : memref<1x3x228x228xf32>
}






