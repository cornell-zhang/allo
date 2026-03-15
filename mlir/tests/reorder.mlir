// RUN: allo-opt %s -split-input-file -transform-interpreter -verify-diagnostics | FileCheck %s

// Regression: reorder a non-contiguous subset of loops in a perfect band.
//
// CHECK-LABEL: func.func @reorder_subset_non_contiguous
// Printed form attaches loop attributes to the closing brace line.
// After reorder, nesting is k (outer) -> j -> i (inner), thus close order is
// i, j, k.
func.func @reorder_subset_non_contiguous(%arg0: memref<4x4x4xi32>) {
  affine.for %i = 0 to 4 {
    affine.for %j = 0 to 4 {
      affine.for %k = 0 to 4 {
        %v = affine.load %arg0[%i, %j, %k] : memref<4x4x4xi32>
        affine.store %v, %arg0[%i, %j, %k] : memref<4x4x4xi32>
        // CHECK: "i"
      } {sym_name = "k"}
      // CHECK-NEXT: "j"
    } {sym_name = "j"}
    // CHECK-NEXT: "k"
  } {sym_name = "i"}
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.op<"builtin.module">) {
    %i = transform.structured.match attributes {sym_name = "i"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"affine.for">
    %k = transform.structured.match attributes {sym_name = "k"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"affine.for">
    %subset = transform.merge_handles %k, %i
      : !transform.op<"affine.for">
    transform.allo.reorder %subset with [1, 0]
      : !transform.op<"affine.for">
    transform.yield
  }
}

// -----

// Regression: selected loop handles are intentionally not in depth order.
// This exercises index mapping from selected loops to full-band positions.
//
// CHECK-LABEL: func.func @reorder_subset_unsorted_handles
// After reorder, nesting is k2 (outer) -> j2 -> i2 (inner), thus close order
// is i2, j2, k2.
func.func @reorder_subset_unsorted_handles(%arg0: memref<4x4x4xi32>) {
  affine.for %i = 0 to 4 {
    affine.for %j = 0 to 4 {
      affine.for %k = 0 to 4 {
        %v = affine.load %arg0[%i, %j, %k] : memref<4x4x4xi32>
        affine.store %v, %arg0[%i, %j, %k] : memref<4x4x4xi32>
        // CHECK: "i2"
      } {sym_name = "k2"}
      // CHECK-NEXT: "j2"
    } {sym_name = "j2"}
    // CHECK-NEXT: "k2"
  } {sym_name = "i2"}
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.op<"builtin.module">) {
    %j = transform.structured.match attributes {sym_name = "j2"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"affine.for">
    %i = transform.structured.match attributes {sym_name = "i2"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"affine.for">
    %k = transform.structured.match attributes {sym_name = "k2"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"affine.for">
    %subset0 = transform.merge_handles %j, %i
      : !transform.op<"affine.for">
    %subset1 = transform.merge_handles %subset0, %k
      : !transform.op<"affine.for">
    transform.allo.reorder %subset1 with [0, 2, 1]
      : !transform.op<"affine.for">
    transform.yield
  }
}

// -----

func.func @reorder_perm_size_mismatch(%arg0: memref<4x4xi32>) {
  affine.for %i = 0 to 4 {
    affine.for %j = 0 to 4 {
      %v = affine.load %arg0[%i, %j] : memref<4x4xi32>
      affine.store %v, %arg0[%i, %j] : memref<4x4xi32>
    } {sym_name = "j"}
  } {sym_name = "i"}
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.op<"builtin.module">) {
    %i = transform.structured.match attributes {sym_name = "i"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"affine.for">
    %j = transform.structured.match attributes {sym_name = "j"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"affine.for">
    %loops = transform.merge_handles %i, %j : !transform.op<"affine.for">
    // expected-error @below {{the size of permutation must match the number of loops}}
    transform.allo.reorder %loops with [0] : !transform.op<"affine.for">
    transform.yield
  }
}

// -----

#map_id = affine_map<(d0) -> (d0)>

func.func @reorder_imperfect_band(%arg0: memref<4x4xi32>) {
  affine.for %i = 0 to 4 {
    %ii = affine.apply #map_id(%i)
    affine.for %j = 0 to 4 {
      %v = affine.load %arg0[%ii, %j] : memref<4x4xi32>
      affine.store %v, %arg0[%ii, %j] : memref<4x4xi32>
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
    transform.allo.reorder %loops with [1, 0] : !transform.op<"affine.for">
    transform.yield
  }
}