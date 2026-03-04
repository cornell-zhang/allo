// RUN: allo-opt %s -split-input-file -transform-interpreter -verify-diagnostics | FileCheck %s

// Regression:
// 1) compute_at should insert affine.if when producer bounds are a subset.
// 2) consumer loop handle must stay reusable after compute_at.
//
// CHECK-LABEL: func.func @compute_at_subset_and_consumer_reusable
func.func @compute_at_subset_and_consumer_reusable(
    %src: memref<16xi32>, %tmp: memref<16xi32>, %dst: memref<16xi32>) {

  affine.for %i = 2 to 6 {
    %v = affine.load %src[%i] : memref<16xi32>
    affine.store %v, %tmp[%i] {sym_name = "producer_store"} : memref<16xi32>
  } {sym_name = "prod_loop"}

  // CHECK: affine.for {{.*}} = 0 to 4 {
  // CHECK-NEXT: affine.for {{.*}} = 0 to 2 {
  // CHECK: affine.if
  affine.for %j = 0 to 8 {
    %u = affine.load %dst[%j] : memref<16xi32>
    affine.store %u, %dst[%j] : memref<16xi32>
  } {sym_name = "cons_loop"}
  // CHECK: "cons_loop.inner"
  // CHECK-NEXT: "cons_loop.outer"
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.op<"builtin.module">) {
    %producer = transform.structured.match attributes {sym_name = "producer_store"} in %root
      : (!transform.op<"builtin.module">) -> !transform.any_op
    %consumer = transform.structured.match attributes {sym_name = "cons_loop"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"affine.for">
    transform.allo.compute_at %producer at %consumer
      : (!transform.any_op, !transform.op<"affine.for">) -> ()
    %outer, %inner = transform.allo.split %consumer with 2
      : (!transform.op<"affine.for">) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// Regression: post-cleanup should do local store-to-load forwarding
//
// CHECK-LABEL: func.func @compute_at_local_store_to_load_forward
func.func @compute_at_local_store_to_load_forward(
    %tmp: memref<8xi32>, %dst: memref<8xi32>) {
  affine.for %i = 0 to 8 {
    %c7 = arith.constant 7 : i32
    affine.store %c7, %tmp[%i] {sym_name = "producer_store_forward"} : memref<8xi32>
  } {sym_name = "prod_loop_forward"}

  // CHECK: affine.for {{.*}} = 0 to 8 {
  affine.for %j = 0 to 8 {
    // CHECK-NOT: affine.load
    %v = affine.load %tmp[%j] : memref<8xi32>
    // CHECK: "producer_store_forward"
    affine.store %v, %dst[%j] : memref<8xi32>
  } {sym_name = "cons_loop_forward"}
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.op<"builtin.module">) {
    %producer = transform.structured.match attributes {sym_name = "producer_store_forward"} in %root
      : (!transform.op<"builtin.module">) -> !transform.any_op
    %consumer = transform.structured.match attributes {sym_name = "cons_loop_forward"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"affine.for">
    transform.allo.compute_at %producer at %consumer
      : (!transform.any_op, !transform.op<"affine.for">) -> ()
    transform.yield
  }
}

// -----

// Producer must be inside an affine.for loop nest.
func.func @compute_at_producer_not_in_loop(%src: memref<8xi32>,
                                             %dst: memref<8xi32>) {
  %v = affine.load %src[0] : memref<8xi32>
  affine.store %v, %dst[0] {sym_name = "top_store"} : memref<8xi32>
  affine.for %i = 0 to 8 {
    %u = affine.load %dst[%i] : memref<8xi32>
    affine.store %u, %dst[%i] : memref<8xi32>
  } {sym_name = "cons_loop_a"}
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.op<"builtin.module">) {
    %producer = transform.structured.match attributes {sym_name = "top_store"} in %root
      : (!transform.op<"builtin.module">) -> !transform.any_op
    %consumer = transform.structured.match attributes {sym_name = "cons_loop_a"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"affine.for">
    // expected-error @below {{producer must be inside an affine.for loop nest}}
    transform.allo.compute_at %producer at %consumer
      : (!transform.any_op, !transform.op<"affine.for">) -> ()
    transform.yield
  }
}

// -----

// In no-dependence move path, producer root must appear before consumer root.
func.func @compute_at_producer_after_consumer(%src: memref<8xi32>,
                                                %tmp: memref<8xi32>,
                                                %cons: memref<8xi32>) {
  affine.for %j = 0 to 8 {
    %u = affine.load %cons[%j] : memref<8xi32>
    affine.store %u, %cons[%j] : memref<8xi32>
  } {sym_name = "cons_loop_d"}

  affine.for %i = 0 to 8 {
    %v = affine.load %src[%i] : memref<8xi32>
    affine.store %v, %tmp[%i] {sym_name = "producer_store_d"} : memref<8xi32>
  } {sym_name = "prod_loop_d"}
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.op<"builtin.module">) {
    %producer = transform.structured.match attributes {sym_name = "producer_store_d"} in %root
      : (!transform.op<"builtin.module">) -> !transform.any_op
    %consumer = transform.structured.match attributes {sym_name = "cons_loop_d"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"affine.for">
    // expected-error @below {{producer root loop must appear before consumer root loop}}
    transform.allo.compute_at %producer at %consumer
      : (!transform.any_op, !transform.op<"affine.for">) -> ()
    transform.yield
  }
}

// -----

// No-dependence move cannot cross side-effecting ops between root loops.
func.func @compute_at_side_effect_between_roots(%src: memref<8xi32>,
                                                  %tmp: memref<8xi32>,
                                                  %mid: memref<8xi32>) {
  affine.for %i = 0 to 8 {
    %v = affine.load %src[%i] : memref<8xi32>
    affine.store %v, %tmp[%i] {sym_name = "producer_store_e"} : memref<8xi32>
  } {sym_name = "prod_loop_e"}

  %c0 = arith.constant 0 : index
  %cst = arith.constant 1 : i32
  memref.store %cst, %tmp[%c0] : memref<8xi32>

  affine.for %j = 0 to 8 {
    %u = affine.load %mid[%j] : memref<8xi32>
    affine.store %u, %mid[%j] : memref<8xi32>
  } {sym_name = "cons_loop_e"}
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.op<"builtin.module">) {
    %producer = transform.structured.match attributes {sym_name = "producer_store_e"} in %root
      : (!transform.op<"builtin.module">) -> !transform.any_op
    %consumer = transform.structured.match attributes {sym_name = "cons_loop_e"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"affine.for">
    // expected-error @below {{cannot move producer across side-effecting operations between producer and consumer roots}}
    transform.allo.compute_at %producer at %consumer
      : (!transform.any_op, !transform.op<"affine.for">) -> ()
    transform.yield
  }
}

// -----

// Unsupported access patterns should fail conservatively when dependence
// analysis cannot conclude.
func.func @compute_at_dependence_analysis_failure(%src: memref<8xi32>,
                                                    %tmp: memref<8xi32>) {
  affine.for %i = 0 to 8 {
    %c0 = arith.constant 0 : index
    %pred = arith.cmpi sgt, %i, %c0 : index
    %idx = arith.select %pred, %i, %c0 : index
    %v = affine.load %src[%i] : memref<8xi32>
    memref.store %v, %tmp[%idx] {sym_name = "producer_store_f"} : memref<8xi32>
  } {sym_name = "prod_loop_f"}

  affine.for %j = 0 to 8 {
    %u = memref.load %tmp[%j] : memref<8xi32>
    memref.store %u, %tmp[%j] : memref<8xi32>
  } {sym_name = "cons_loop_f"}
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.op<"builtin.module">) {
    %producer = transform.structured.match attributes {sym_name = "producer_store_f"} in %root
      : (!transform.op<"builtin.module">) -> !transform.any_op
    %consumer = transform.structured.match attributes {sym_name = "cons_loop_f"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"affine.for">
    // expected-error @below {{dependence analysis failed; refusing compute_at}}
    transform.allo.compute_at %producer at %consumer
      : (!transform.any_op, !transform.op<"affine.for">) -> ()
    transform.yield
  }
}

// -----

// Consumer handle must resolve to affine.for.
func.func @compute_at_consumer_not_affine(%src: memref<8xi32>,
                                            %dst: memref<8xi32>) {
  affine.for %i = 0 to 8 {
    %v = affine.load %src[%i] : memref<8xi32>
    affine.store %v, %dst[%i] {sym_name = "producer_store_b"} : memref<8xi32>
  } {sym_name = "prod_loop_b"}
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %j = %c0 to %c8 step %c1 {
    %u = memref.load %dst[%j] : memref<8xi32>
    memref.store %u, %dst[%j] : memref<8xi32>
  } {sym_name = "cons_loop_b"}
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.op<"builtin.module">) {
    %producer = transform.structured.match attributes {sym_name = "producer_store_b"} in %root
      : (!transform.op<"builtin.module">) -> !transform.any_op
    %consumer = transform.structured.match attributes {sym_name = "cons_loop_b"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"scf.for">
    // expected-error @below {{expected consumer_loop to resolve to an affine.for}}
    transform.allo.compute_at %producer at %consumer
      : (!transform.any_op, !transform.op<"scf.for">) -> ()
    transform.yield
  }
}

// -----

// Producer and consumer must be in different root loop nests.
func.func @compute_at_same_root_loop_nest(%src: memref<8xi32>,
                                            %dst: memref<8xi32>) {
  affine.for %i = 0 to 8 {
    %v = affine.load %src[%i] : memref<8xi32>
    affine.store %v, %dst[%i] {sym_name = "producer_store_c"} : memref<8xi32>
  } {sym_name = "cons_loop_c"}
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.op<"builtin.module">) {
    %producer = transform.structured.match attributes {sym_name = "producer_store_c"} in %root
      : (!transform.op<"builtin.module">) -> !transform.any_op
    %consumer = transform.structured.match attributes {sym_name = "cons_loop_c"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"affine.for">
    // expected-error @below {{producer and consumer must belong to different root loop nests}}
    transform.allo.compute_at %producer at %consumer
      : (!transform.any_op, !transform.op<"affine.for">) -> ()
    transform.yield
  }
}
