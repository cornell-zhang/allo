// RUN: allo-opt %s -split-input-file -transform-interpreter -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @raise_for_with_minmax
func.func @raise_for_with_minmax(%arg0: memref<64xf32>, %arg1: memref<64xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %c48 = arith.constant 48 : index
  %c16 = arith.constant 16 : index
  %c8 = arith.constant 8 : index

  %lb = arith.maxsi %c0, %c8 : index
  %ub = arith.minsi %c48, %c32 : index
  // CHECK: affine.for
  // CHECK-NOT: scf.for
  scf.for %i = %lb to %ub step %c1 {
    %j = arith.addi %i, %c16 : index
    // CHECK: affine.load
    // CHECK-NOT: memref.load
    %a = memref.load %arg0[%j] : memref<64xf32>
    %b = memref.load %arg1[%i] : memref<64xf32>
    %sum = arith.addf %a, %b : f32
    // CHECK: affine.store
    // CHECK-NOT: memref.store
    memref.store %sum, %arg0[%j] : memref<64xf32>
  } { sym_name = "loop" }
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.op<"builtin.module">) {
    %loop = transform.structured.match attributes { sym_name = "loop" } in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"scf.for">
    %raised = transform.allo.raise_to_affine %loop
      : !transform.op<"scf.for"> -> !transform.op<"affine.for">
    transform.yield
  }
}

// -----

// CHECK-LABEL: func.func @raise_parallel_no_reduction
func.func @raise_parallel_no_reduction(%arg0: memref<64x64xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index

  %lb0 = arith.maxsi %c0, %c1 : index
  %ub0 = arith.minsi %c32, %c16 : index
  %lb1 = arith.maxsi %c0, %c1 : index
  %ub1 = arith.minsi %c32, %c16 : index
  // CHECK: affine.parallel
  // CHECK-NOT: scf.parallel
  scf.parallel (%i, %j) = (%lb0, %lb1) to (%ub0, %ub1) step (%c1, %c1) {
    // CHECK: affine.load
    // CHECK-NOT: memref.load
    %v = memref.load %arg0[%i, %j] : memref<64x64xf32>
    // CHECK: affine.store
    // CHECK-NOT: memref.store
    memref.store %v, %arg0[%i, %j] : memref<64x64xf32>
    scf.reduce
  } { sym_name = "ploop" }
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.op<"builtin.module">) {
    %loop = transform.structured.match attributes { sym_name = "ploop" } in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"scf.parallel">
    %raised = transform.allo.raise_to_affine %loop
      : !transform.op<"scf.parallel"> -> !transform.op<"affine.parallel">
    transform.yield
  }
}

// -----

// CHECK-LABEL: func.func @raise_for_with_select_minmax
func.func @raise_for_with_select_minmax(%arg0: memref<64xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %c20 = arith.constant 20 : index
  %c40 = arith.constant 40 : index

  %cmp_lb = arith.cmpi sge, %c20, %c10 : index
  %lb = arith.select %cmp_lb, %c20, %c10 : index
  %cmp_ub = arith.cmpi sle, %c40, %c20 : index
  %ub = arith.select %cmp_ub, %c40, %c20 : index

  // CHECK: affine.for
  // CHECK-NOT: scf.for
  scf.for %i = %lb to %ub step %c1 {
    // CHECK: affine.load
    // CHECK-NOT: memref.load
    %v = memref.load %arg0[%i] : memref<64xf32>
    // CHECK: affine.store
    // CHECK-NOT: mmeref.store
    memref.store %v, %arg0[%i] : memref<64xf32>
  } { sym_name = "sloop" }
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.op<"builtin.module">) {
    %loop = transform.structured.match attributes { sym_name = "sloop" } in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"scf.for">
    %raised = transform.allo.raise_to_affine %loop
      : !transform.op<"scf.for"> -> !transform.op<"affine.for">
    transform.yield
  }
}

// -----

// CHECK-LABEL: func.func @raise_multiple_loops
func.func @raise_multiple_loops(%arg0: memref<10x10xf32>) {
  %c0 = arith.constant 0 : index
  %c5 = arith.constant 5 : index
  %c1 = arith.constant 1 : index
  // CHECK: affine.for
  scf.for %arg1 = %c0 to %c5 step %c1 {
    // CHECK: affine.for
    scf.for %arg2 = %c1 to %c5 step %c1 {
      // CHECK: affine.for
      scf.for %arg3 = %c1 to %c5 step %c1 {
        // CHECK: affine.load
        %0 = memref.load %arg0[%arg1, %arg2] : memref<10x10xf32>
        %1 = memref.load %arg0[%arg1, %arg3] : memref<10x10xf32>
        %2 = arith.addf %0, %1 : f32
        memref.store %2, %arg0[%arg1, %arg3] : memref<10x10xf32>
      } { sym_name = "loop_0_0_0" }
    } { sym_name = "loop_0_0" }
  } { sym_name = "loop_0" }
  func.return
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.op<"builtin.module">) {
    %loops = transform.structured.match ops{["scf.for"]} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"scf.for">
    %raised = transform.allo.raise_to_affine %loops
      : !transform.op<"scf.for"> -> !transform.op<"affine.for">
    transform.yield
  }
}

// -----

func.func @raise_for_non_const_step(%arg0: memref<8xi32>, %step: index) {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  // expected-error @below {{step is not a constant positive integer}}
  scf.for %i = %c0 to %c8 step %step {
    %v = memref.load %arg0[%i] : memref<8xi32>
    memref.store %v, %arg0[%i] : memref<8xi32>
  } {sym_name = "for_bad_step"}
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.op<"builtin.module">) {
    %loop = transform.structured.match attributes {sym_name = "for_bad_step"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"scf.for">
    %raised = transform.allo.raise_to_affine %loop
      : !transform.op<"scf.for"> -> !transform.op<"affine.for">
    transform.yield
  }
}

// -----

func.func @raise_parallel_with_reduction() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c0_i32 = arith.constant 0 : i32
  // expected-error @below {{parallel reduction is not supported yet}}
  %sum = scf.parallel (%i) = (%c0) to (%c8) step (%c1) init (%c0_i32) -> (i32) {
    %one = arith.constant 1 : i32
    scf.reduce(%one : i32) {
    ^bb0(%lhs: i32, %rhs: i32):
      %add = arith.addi %lhs, %rhs : i32
      scf.reduce.return %add : i32
    }
  } {sym_name = "parallel_reduce"}
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.op<"builtin.module">) {
    %loop = transform.structured.match attributes {sym_name = "parallel_reduce"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"scf.parallel">
    %raised = transform.allo.raise_to_affine %loop
      : !transform.op<"scf.parallel"> -> !transform.op<"affine.parallel">
    transform.yield
  }
}
