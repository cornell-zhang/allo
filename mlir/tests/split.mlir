// RUN: allo-opt %s -split-input-file -transform-interpreter -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @split_affine_loop
func.func @split_affine_loop(%arg0: memref<8xi32>) {
  // CHECK: affine.for {{.*}} = 0 to 4
  affine.for %i = 0 to 8 {
    // CHECK-NOT: affine.apply
    // CHECK: affine.for {{.*}} = 0 to 2
    %v = affine.load %arg0[%i] : memref<8xi32>
    affine.store %v, %arg0[%i] : memref<8xi32>
  } {sym_name = "aloop"}
  // CHECK: "aloop.inner"
  // CHECK: "aloop.outer"
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.op<"builtin.module">) {
    %loop = transform.structured.match attributes {sym_name = "aloop"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"affine.for">
    %outer, %inner = transform.allo.split %loop with 2
      : (!transform.op<"affine.for">) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// CHECK-LABEL: func.func @split_scf_loop
func.func @split_scf_loop(%arg0: memref<8xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  // CHECK: %c2 = arith.constant 2 : index
  // CHECK: scf.for {{.*}} step
  scf.for %i = %c0 to %c8 step %c1 {
    // CHECK: arith.minsi
    // CHECK: scf.for {{.*}} to {{.*}} step %c1
    %v = memref.load %arg0[%i] : memref<8xi32>
    memref.store %v, %arg0[%i] : memref<8xi32>
  } {sym_name = "sloop"}
  // CHECK: "sloop.inner"
  // CHECK: "sloop.outer"
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.op<"builtin.module">) {
    %loop = transform.structured.match attributes {sym_name = "sloop"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"scf.for">
    %outer, %inner = transform.allo.split %loop with 2
      : (!transform.op<"scf.for">) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

func.func @split_non_positive_factor(%arg0: memref<8xi32>) {
  affine.for %i = 0 to 8 {
    %v = affine.load %arg0[%i] : memref<8xi32>
    affine.store %v, %arg0[%i] : memref<8xi32>
  } {sym_name = "a_loop"}
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.op<"builtin.module">) {
    %loop = transform.structured.match attributes {sym_name = "a_loop"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"affine.for">
    // expected-error @below {{split factor must be positive}}
    %outer, %inner = transform.allo.split %loop with 0
      : (!transform.op<"affine.for">) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

func.func @split_factor_too_large_scf(%arg0: memref<8xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  // expected-error @below {{split factor is larger than or equal to the loop range}}
  scf.for %i = %c0 to %c8 step %c1 {
    %v = memref.load %arg0[%i] : memref<8xi32>
    memref.store %v, %arg0[%i] : memref<8xi32>
  } {sym_name = "s_loop"}
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.op<"builtin.module">) {
    %loop = transform.structured.match attributes {sym_name = "s_loop"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"scf.for">
    %outer, %inner = transform.allo.split %loop with 8
      : (!transform.op<"scf.for">) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}