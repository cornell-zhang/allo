// RUN: allo-opt %s -split-input-file -transform-interpreter -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @inner
// CHECK: "inner"
// CHECK-LABEL: func.func @outer
// CHECK: call @inner{{.*}} "inner::call"
// CHECK: "outer"
// CHECK-LABEL: func.func @outline_loops
// CHECK: call @outer{{.*}} "outer::call"
func.func @outline_loops(%arg0: memref<4x4xf32>, %arg1: memref<4x4xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    scf.for %j = %c0 to %c4 step %c1 {
      %a = memref.load %arg0[%i, %j] : memref<4x4xf32>
      %b = memref.load %arg1[%i, %j] : memref<4x4xf32>
      %c = arith.addf %a, %b : f32
      memref.store %c, %arg0[%i, %j] : memref<4x4xf32>
    } { sym_name = "inner" }
  } { sym_name = "outer"}
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main (%root: !transform.op<"builtin.module">) {
    %outer = transform.structured.match attributes {sym_name = "outer"} in %root : (!transform.op<"builtin.module">) -> !transform.op<"scf.for">
    %inner = transform.structured.match attributes {sym_name = "inner"} in %root : (!transform.op<"builtin.module">) -> !transform.op<"scf.for">
    %inner_k, %inner_c = transform.allo.outline %inner to "inner": (!transform.op<"scf.for">) -> (!transform.op<"func.func">, !transform.op<"func.call">)
    %outer_k, %outer_c = transform.allo.outline %outer to "outer": (!transform.op<"scf.for">) -> (!transform.op<"func.func">, !transform.op<"func.call">)
    transform.yield
  }
}

// -----

// CHECK-LABEL: func.func @scalar_load
// CHECK: "scalar_load"
// CHECK-LABEL: func.func @outline_no_region_op
// CHECK: call @scalar_load{{.*}} "scalar_load::call"
func.func @outline_no_region_op(%arg0: memref<4xi32>, %arg1: memref<4xi32>) {
  %c0 = arith.constant 0 : index
  %c1_i32 = arith.constant 1 : i32
  %v = memref.load %arg0[%c0] {sym_name = "scalar_load"} : memref<4xi32>
  %sum = arith.addi %v, %c1_i32 : i32
  memref.store %sum, %arg1[%c0] : memref<4xi32>
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main (%root: !transform.op<"builtin.module">) {
    %load = transform.structured.match attributes {sym_name = "scalar_load"} in %root : (!transform.op<"builtin.module">) -> !transform.op<"memref.load">
    %load_k, %load_c = transform.allo.outline %load to "scalar_load": (!transform.op<"memref.load">) -> (!transform.op<"func.func">, !transform.op<"func.call">)
    transform.yield
  }
}

// -----

// CHECK-LABEL: func.func @exec_region
// CHECK: scf.execute_region
// CHECK-LABEL: func.func @outline_single_region_op
// CHECK: call @exec_region
func.func @outline_single_region_op(%arg0: i32) -> (i32) {
  %v = scf.execute_region -> i32 {
    %c1_i32 = arith.constant 1 : i32
    %add = arith.addi %arg0, %c1_i32 : i32
    scf.yield %add : i32
  } {sym_name = "exec_region"}
  func.return %v : i32
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main (%root: !transform.op<"builtin.module">) {
    %exec = transform.structured.match attributes {sym_name = "exec_region"} in %root : (!transform.op<"builtin.module">) -> !transform.op<"scf.execute_region">
    %exec_k, %exec_c = transform.allo.outline %exec to "exec_region": (!transform.op<"scf.execute_region">) -> (!transform.op<"func.func">, !transform.op<"func.call">)
    transform.yield
  }
}

// -----

func.func @outline_multi_region_op() {
  %cond = arith.constant true
  %c0_i32 = arith.constant 0 : i32
  // expected-error @below {{expected target operation to have at most one region}}
  scf.if %cond {
    %x = arith.addi %c0_i32, %c0_i32 : i32
  } else {
    %y = arith.addi %c0_i32, %c0_i32 : i32
  } {sym_name = "if_bad"}
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.op<"builtin.module">) {
    %if = transform.structured.match attributes {sym_name = "if_bad"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"scf.if">
    %k, %c = transform.allo.outline %if to "if_kernel"
      : (!transform.op<"scf.if">) -> (!transform.op<"func.func">, !transform.op<"func.call">)
    transform.yield
  }
}
