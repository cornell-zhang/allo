// RUN: allo-opt %s -split-input-file -transform-interpreter -verify-diagnostics | FileCheck %s

// Regression: flatten should accept unordered loop handles and still flatten
// from outermost to innermost selected loop.
//
// CHECK-LABEL: func.func @flatten_affine_unsorted_handles
func.func @flatten_affine_unsorted_handles(%arg0: memref<4x5x6xi32>) {
  // CHECK: affine.for {{.*}} = 0 to 120 {
  affine.for %i = 0 to 4 {
    affine.for %j = 0 to 5 {
      affine.for %k = 0 to 6 {
        %v = affine.load %arg0[%i, %j, %k] : memref<4x5x6xi32>
        affine.store %v, %arg0[%i, %j, %k] : memref<4x5x6xi32>
      } {sym_name = "k_f"}
    } {sym_name = "j_f"}
  } {sym_name = "i_f"}
  // CHECK: "j_f::flat"
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.op<"builtin.module">) {
    %j = transform.structured.match attributes {sym_name = "j_f"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"affine.for">
    %i = transform.structured.match attributes {sym_name = "i_f"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"affine.for">
    %k = transform.structured.match attributes {sym_name = "k_f"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"affine.for">
    %tmp = transform.merge_handles %j, %i : !transform.op<"affine.for">
    %loops = transform.merge_handles %tmp, %k : !transform.op<"affine.for">
    %flat = transform.allo.flatten %loops
      : !transform.op<"affine.for"> -> !transform.op<"affine.for">
    transform.yield
  }
}

// -----

// Regression: selected loops may be non-contiguous in the handle list; if the
// range is perfectly nested, flatten should coalesce the full contiguous range.
//
// CHECK-LABEL: func.func @flatten_affine_non_contiguous_selection
func.func @flatten_affine_non_contiguous_selection(%arg0: memref<4x5x6xi32>) {
  // CHECK: affine.for {{.*}} = 0 to 120 {
  affine.for %i = 0 to 4 {
    affine.for %j = 0 to 5 {
      affine.for %k = 0 to 6 {
        %v = affine.load %arg0[%i, %j, %k] : memref<4x5x6xi32>
        affine.store %v, %arg0[%i, %j, %k] : memref<4x5x6xi32>
      } {sym_name = "k_nc"}
    } {sym_name = "j_nc"}
  } {sym_name = "i_nc"}
  // CHECK: "k_nc::flat"
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.op<"builtin.module">) {
    %k = transform.structured.match attributes {sym_name = "k_nc"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"affine.for">
    %i = transform.structured.match attributes {sym_name = "i_nc"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"affine.for">
    %loops = transform.merge_handles %k, %i : !transform.op<"affine.for">
    %flat = transform.allo.flatten %loops
      : !transform.op<"affine.for"> -> !transform.op<"affine.for">
    transform.yield
  }
}

// -----

#map_id = affine_map<(d0) -> (d0)>

func.func @flatten_imperfect_nest(%arg0: memref<4x5xi32>) {
  affine.for %i = 0 to 4 {
    %ii = affine.apply #map_id(%i)
    affine.for %j = 0 to 5 {
      %v = affine.load %arg0[%ii, %j] : memref<4x5xi32>
      affine.store %v, %arg0[%ii, %j] : memref<4x5xi32>
    } {sym_name = "j_bad"}
  } {sym_name = "i_bad"}
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.op<"builtin.module">) {
    %i = transform.structured.match attributes {sym_name = "i_bad"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"affine.for">
    %j = transform.structured.match attributes {sym_name = "j_bad"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"affine.for">
    %loops = transform.merge_handles %i, %j : !transform.op<"affine.for">
    // expected-error @below {{loops must be in the same perfectly nested loop band}}
    %flat = transform.allo.flatten %loops
      : !transform.op<"affine.for"> -> !transform.op<"affine.for">
    transform.yield
  }
}

// -----

func.func @flatten_non_normalized_loops(%arg0: memref<6x5xi32>) {
  affine.for %i = 1 to 6 {
    affine.for %j = 0 to 5 step 2 {
      %v = affine.load %arg0[%i, %j] : memref<6x5xi32>
      affine.store %v, %arg0[%i, %j] : memref<6x5xi32>
    } {sym_name = "j_norm"}
  } {sym_name = "i_norm"}
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.op<"builtin.module">) {
    %j = transform.structured.match attributes {sym_name = "j_norm"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"affine.for">
    %i = transform.structured.match attributes {sym_name = "i_norm"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"affine.for">
    %loops = transform.merge_handles %j, %i : !transform.op<"affine.for">
    // expected-error @below {{flatten requires normalized affine.for loops with step=1, constant lower bound=0 and constant upper bound}}
    %flat = transform.allo.flatten %loops
      : !transform.op<"affine.for"> -> !transform.op<"affine.for">
    transform.yield
  }
}
