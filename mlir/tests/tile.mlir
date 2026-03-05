// RUN: allo-opt %s -split-input-file -transform-interpreter -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @tile_affine_loop
func.func @tile_affine_loop(%arg0: memref<8xi32>) {
  // CHECK: affine.for {{.*}} = 0 to 4
  affine.for %i = 0 to 8 {
    // CHECK: affine.for {{.*}} = 0 to 2
    %v = affine.load %arg0[%i] : memref<8xi32>
    affine.store %v, %arg0[%i] : memref<8xi32>
  } {sym_name = "aloop"}
  // CHECK: "aloop::point"
  // CHECK: "aloop::tile"
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.op<"builtin.module">) {
    %loop = transform.structured.match attributes {sym_name = "aloop"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"affine.for">
    %tile_loop, %point_loop = transform.allo.tile %loop with [2]
      : (!transform.op<"affine.for">) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// Regression: factors follow handle order, then are reordered with loops by
// depth before tiling. Handles are merged as (j, i), factors are [5, 3], so
// final mapping must be i->3, j->5.
//
// CHECK-LABEL: func.func @tile_affine_perfect_unsorted_handles
func.func @tile_affine_perfect_unsorted_handles(
    %arg0: memref<15x30xi32>) {
  // CHECK: affine.for {{.*}} = 0 to 5
  affine.for %i = 0 to 15 {
    // CHECK-NOT: affine.apply
    // CHECK: affine.for {{.*}} = 0 to 6
    affine.for %j = 0 to 30 {
      // CHECK-NOT: affine.apply
      // CHECK: affine.for {{.*}} = 0 to 3
      // CHECK-NOT: affine.apply
      // CHECK: affine.for {{.*}} = 0 to 5
      %v = affine.load %arg0[%i, %j] : memref<15x30xi32>
      affine.store %v, %arg0[%i, %j] : memref<15x30xi32>
    } {sym_name = "j_ap"}
    // CHECK: "j_ap::point"
  } {sym_name = "i_ap"}
  // CHECK: "i_ap::point"
  // CHECK: "j_ap::tile"
  // CHECK: "i_ap::tile"
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.op<"builtin.module">) {
    %i = transform.structured.match attributes {sym_name = "i_ap"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"affine.for">
    %j = transform.structured.match attributes {sym_name = "j_ap"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"affine.for">
    %loops = transform.merge_handles %j, %i
      : !transform.op<"affine.for">
    %tile_loop, %point_loop = transform.allo.tile %loops with [5, 3]
      : (!transform.op<"affine.for">) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

#map_id = affine_map<(d0) -> (d0)>

// Regression: same semantics for imperfect affine nests.
//
// CHECK-LABEL: func.func @tile_affine_imperfect_unsorted_handles
func.func @tile_affine_imperfect_unsorted_handles(
    %arg0: memref<15x30xi32>) {
  // CHECK: affine.for {{.*}} = 0 to 15 step 3
  affine.for %i = 0 to 15 {
    %ii = affine.apply #map_id(%i)
    // CHECK: affine.for {{.*}} = 0 to 30 step 5
    affine.for %j = 0 to 30 {
      %v = affine.load %arg0[%ii, %j] : memref<15x30xi32>
      affine.store %v, %arg0[%ii, %j] : memref<15x30xi32>
    } {sym_name = "j_ai"}
  } {sym_name = "i_ai"}
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.op<"builtin.module">) {
    %i = transform.structured.match attributes {sym_name = "i_ai"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"affine.for">
    %j = transform.structured.match attributes {sym_name = "j_ai"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"affine.for">
    %loops = transform.merge_handles %j, %i
      : !transform.op<"affine.for">
    %tile_loop, %point_loop = transform.allo.tile %loops with [5, 3]
      : (!transform.op<"affine.for">) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// Regression: perfect scf nest with unsorted handles. Final mapping must be
// i->3, j->5.
//
// CHECK-LABEL: func.func @tile_scf_perfect_unsorted_handles
func.func @tile_scf_perfect_unsorted_handles(%arg0: memref<15x30xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c15 = arith.constant 15 : index
  %c30 = arith.constant 30 : index
  // CHECK-DAG: %[[C3:.*]] = arith.constant 3 : index
  // CHECK-DAG: %[[C5:.*]] = arith.constant 5 : index
  // CHECK: %[[S3:.*]] = arith.muli %c1, %[[C3]] : index
  // CHECK: scf.for {{.*}} step %[[S3]]
  scf.for %i = %c0 to %c15 step %c1 {
    // CHECK: %[[S5:.*]] = arith.muli %c1, %[[C5]] : index
    // CHECK: scf.for {{.*}} step %[[S5]]
    scf.for %j = %c0 to %c30 step %c1 {
      %v = memref.load %arg0[%i, %j] : memref<15x30xi32>
      memref.store %v, %arg0[%i, %j] : memref<15x30xi32>
    } {sym_name = "j_sp"}
  } {sym_name = "i_sp"}
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.op<"builtin.module">) {
    %i = transform.structured.match attributes {sym_name = "i_sp"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"scf.for">
    %j = transform.structured.match attributes {sym_name = "j_sp"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"scf.for">
    %loops = transform.merge_handles %j, %i
      : !transform.op<"scf.for">
    %tile_loop, %point_loop = transform.allo.tile %loops with [5, 3]
      : (!transform.op<"scf.for">) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// Regression: same semantics for imperfect scf nests.
//
// CHECK-LABEL: func.func @tile_scf_imperfect_unsorted_handles
func.func @tile_scf_imperfect_unsorted_handles(%arg0: memref<15x30xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c15 = arith.constant 15 : index
  %c30 = arith.constant 30 : index
  // CHECK-DAG: %[[C3:.*]] = arith.constant 3 : index
  // CHECK-DAG: %[[C5:.*]] = arith.constant 5 : index
  // CHECK: %[[S3:.*]] = arith.muli %c1, %[[C3]] : index
  // CHECK: scf.for {{.*}} step %[[S3]]
  scf.for %i = %c0 to %c15 step %c1 {
    %ii = arith.addi %i, %c0 : index
    // CHECK: %[[S5:.*]] = arith.muli %c1, %[[C5]] : index
    // CHECK: scf.for {{.*}} step %[[S5]]
    scf.for %j = %c0 to %c30 step %c1 {
      %v = memref.load %arg0[%ii, %j] : memref<15x30xi32>
      memref.store %v, %arg0[%ii, %j] : memref<15x30xi32>
    } {sym_name = "j_si"}
  } {sym_name = "i_si"}
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.op<"builtin.module">) {
    %i = transform.structured.match attributes {sym_name = "i_si"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"scf.for">
    %j = transform.structured.match attributes {sym_name = "j_si"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"scf.for">
    %loops = transform.merge_handles %j, %i
      : !transform.op<"scf.for">
    %tile_loop, %point_loop = transform.allo.tile %loops with [5, 3]
      : (!transform.op<"scf.for">) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

func.func @tile_mixed_loop_dialects(%arg0: memref<8x8xi32>) {
  affine.for %i = 0 to 8 {
    %v0 = affine.load %arg0[%i, %i] : memref<8x8xi32>
    affine.store %v0, %arg0[%i, %i] : memref<8x8xi32>
  } {sym_name = "a_mix"}

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %j = %c0 to %c8 step %c1 {
    %v1 = memref.load %arg0[%j, %j] : memref<8x8xi32>
    memref.store %v1, %arg0[%j, %j] : memref<8x8xi32>
  } {sym_name = "s_mix"}
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.op<"builtin.module">) {
    %loops = transform.structured.match ops{["affine.for", "scf.for"]} in %root
      : (!transform.op<"builtin.module">) -> !transform.any_op
    // expected-error @below {{cannot mix affine.for and scf.for loops in the same tiling}}
    %tile, %point = transform.allo.tile %loops with [2, 2]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

func.func @tile_factor_count_mismatch(%arg0: memref<8x8xi32>) {
  affine.for %i = 0 to 8 {
    affine.for %j = 0 to 8 {
      %v = affine.load %arg0[%i, %j] : memref<8x8xi32>
      affine.store %v, %arg0[%i, %j] : memref<8x8xi32>
    } {sym_name = "j_cnt"}
  } {sym_name = "i_cnt"}
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.op<"builtin.module">) {
    %i = transform.structured.match attributes {sym_name = "i_cnt"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"affine.for">
    %j = transform.structured.match attributes {sym_name = "j_cnt"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"affine.for">
    %loops = transform.merge_handles %i, %j : !transform.op<"affine.for">
    // expected-error @below {{number of tile factors must match the number of loops}}
    %tile, %point = transform.allo.tile %loops with [2]
      : (!transform.op<"affine.for">) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

func.func @tile_affine_not_same_nest(%arg0: memref<8xi32>) {
  affine.for %i = 0 to 8 {
    %v0 = affine.load %arg0[%i] : memref<8xi32>
    affine.store %v0, %arg0[%i] : memref<8xi32>
  } {sym_name = "i_bad"}
  affine.for %j = 0 to 8 {
    %v1 = affine.load %arg0[%j] : memref<8xi32>
    affine.store %v1, %arg0[%j] : memref<8xi32>
  } {sym_name = "j_bad"}
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.op<"builtin.module">) {
    %i = transform.structured.match attributes {sym_name = "i_bad"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"affine.for">
    %j = transform.structured.match attributes {sym_name = "j_bad"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"affine.for">
    %loops = transform.merge_handles %i, %j : !transform.op<"affine.for">
    // expected-error @below {{affine loops must be unique and in the same loop nest}}
    %tile, %point = transform.allo.tile %loops with [2, 2]
      : (!transform.op<"affine.for">) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
