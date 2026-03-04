// RUN: allo-opt %s -split-input-file -transform-interpreter -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @reuse_at_basic
// CHECK: memref.alloc() {sym_name = "in_buf.reuse"} : memref<3xi32>
// CHECK: affine.load %{{.*}}[0] : memref<3xi32>
func.func @reuse_at_basic(%out: memref<8x6xi32>) {
  %in_buf = memref.alloc() {sym_name = "in_buf"} : memref<8x8xi32>
  affine.for %y = 0 to 8 {
    affine.for %x = 0 to 6 {
      %a0 = affine.load %in_buf[%y, %x] : memref<8x8xi32>
      %a1 = affine.load %in_buf[%y, %x + 1] : memref<8x8xi32>
      %a2 = affine.load %in_buf[%y, %x + 2] : memref<8x8xi32>
      %t0 = arith.addi %a0, %a1 : i32
      %t1 = arith.addi %t0, %a2 : i32
      affine.store %t1, %out[%y, %x] : memref<8x6xi32>
    } {sym_name = "x"}
  } {sym_name = "y"}
  memref.dealloc %in_buf : memref<8x8xi32>
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.op<"builtin.module">) {
    %alloc = transform.structured.match ops{["memref.alloc"]} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"memref.alloc">
    %target = transform.allo.match_value 0 of %alloc
      : !transform.op<"memref.alloc"> -> !transform.any_value
    %axis = transform.structured.match attributes {sym_name = "x"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"affine.for">
    %reuse = transform.allo.reuse_at %target at %axis
      : (!transform.any_value, !transform.op<"affine.for">) -> !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: func.func @reuse_at_affine_apply_chain
// CHECK: memref.alloc() {sym_name = "chain_buf.reuse"} : memref<3xi32>
// CHECK: affine.load %{{.*}}[0] : memref<3xi32>
func.func @reuse_at_affine_apply_chain(%out: memref<8x6xi32>) {
  %chain_buf = memref.alloc() {sym_name = "chain_buf"} : memref<8x10xi32>
  affine.for %y = 0 to 8 {
    affine.for %x = 0 to 6 {
      %x_shift = affine.apply affine_map<(d0) -> (d0 + 1)>(%x)
      %a0 = affine.load %chain_buf[%y, %x_shift] : memref<8x10xi32>
      %a1 = affine.load %chain_buf[%y, %x_shift + 1] : memref<8x10xi32>
      %a2 = affine.load %chain_buf[%y, %x_shift + 2] : memref<8x10xi32>
      %t0 = arith.addi %a0, %a1 : i32
      %t1 = arith.addi %t0, %a2 : i32
      affine.store %t1, %out[%y, %x] : memref<8x6xi32>
    } {sym_name = "x"}
  } {sym_name = "y"}
  memref.dealloc %chain_buf : memref<8x10xi32>
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.op<"builtin.module">) {
    %alloc = transform.structured.match ops{["memref.alloc"]} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"memref.alloc">
    %target = transform.allo.match_value 0 of %alloc
      : !transform.op<"memref.alloc"> -> !transform.any_value
    %axis = transform.structured.match attributes {sym_name = "x"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"affine.for">
    %reuse = transform.allo.reuse_at %target at %axis
      : (!transform.any_value, !transform.op<"affine.for">) -> !transform.any_op
    transform.yield
  }
}

// -----

func.func @reuse_at_reduction_not_supported(%out: memref<8x6xi32>) {
  %in_buf = memref.alloc() {sym_name = "reduce_buf"} : memref<8x8xi32>
  affine.for %y = 0 to 8 {
    affine.for %x = 0 to 6 {
      affine.for %r = 0 to 3 {
        %v = affine.load %in_buf[%y, %x + %r] : memref<8x8xi32>
        affine.store %v, %out[%y, %x] : memref<8x6xi32>
      } {sym_name = "r"}
    } {sym_name = "x"}
  } {sym_name = "y"}
  memref.dealloc %in_buf : memref<8x8xi32>
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.op<"builtin.module">) {
    %alloc = transform.structured.match ops{["memref.alloc"]} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"memref.alloc">
    %target = transform.allo.match_value 0 of %alloc
      : !transform.op<"memref.alloc"> -> !transform.any_value
    %axis = transform.structured.match attributes {sym_name = "x"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"affine.for">
    // expected-error @below {{reduction reuse not fully implemented}}
    %reuse = transform.allo.reuse_at %target at %axis
      : (!transform.any_value, !transform.op<"affine.for">) -> !transform.any_op
    transform.yield
  }
}

// -----

func.func @reuse_at_ignore_unrelated_store(%out: memref<8x6xi32>) {
  %in_buf = memref.alloc() {sym_name = "classify_buf"} : memref<8x8xi32>
  %scratch = memref.alloc() : memref<3xi32>
  %c0 = arith.constant 0 : i32
  affine.for %y = 0 to 8 {
    affine.for %x = 0 to 6 {
      affine.for %r = 0 to 3 {
        %v = affine.load %in_buf[%y, %x + %r] : memref<8x8xi32>
        affine.store %c0, %scratch[%r] : memref<3xi32>
      } {sym_name = "r"}
      affine.store %c0, %out[%y, %x] : memref<8x6xi32>
    } {sym_name = "x"}
  } {sym_name = "y"}
  memref.dealloc %scratch : memref<3xi32>
  memref.dealloc %in_buf : memref<8x8xi32>
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.op<"builtin.module">) {
    %alloc = transform.structured.match ops{["memref.alloc"]} attributes {sym_name = "classify_buf"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"memref.alloc">
    %target = transform.allo.match_value 0 of %alloc
      : !transform.op<"memref.alloc"> -> !transform.any_value
    %axis = transform.structured.match attributes {sym_name = "r"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"affine.for">
    // expected-error @below {{selected axis loop is classified as a reduction loop}}
    %reuse = transform.allo.reuse_at %target at %axis
      : (!transform.any_value, !transform.op<"affine.for">) -> !transform.any_op
    transform.yield
  }
}

