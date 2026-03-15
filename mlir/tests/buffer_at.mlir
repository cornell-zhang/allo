// RUN: allo-opt %s -split-input-file -transform-interpreter -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @buffer_at_basic
func.func @buffer_at_basic() {
  %tmp = memref.alloc() {sym_name = "tmp"} : memref<8x8xi32>
  %c1 = arith.constant 1 : i32
  affine.for %i = 0 to 8 {
    // CHECK: memref.alloc(){{.*}} : memref<1x8xi32>
    affine.for %j = 0 to 8 {
      affine.store %c1, %tmp[%i, %j] : memref<8x8xi32>
    } {sym_name = "j"}
    // CHECK: affine.store %{{.*}}, %{{.*}}[0, %{{.*}}] : memref<1x8xi32>
    // CHECK: affine.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<8x8xi32>
  } {sym_name = "i"}
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    %alloc = transform.structured.match attributes {sym_name = "tmp"} in %root
      : (!transform.any_op) -> !transform.any_op
    %target = transform.allo.match_value 0 of %alloc kind 2
      : !transform.any_op -> !transform.any_value
    %axis = transform.structured.match attributes {sym_name = "i"} in %root
      :(!transform.any_op) -> !transform.any_op
    %local = transform.allo.buffer_at %target at %axis
      : (!transform.any_value, !transform.any_op) -> !transform.any_value
    transform.yield
  }
}

// -----

// CHECK-LABEL: func.func @buffer_at_offset
func.func @buffer_at_offset() {
  %tmp = memref.alloc() {sym_name = "tmp"} : memref<8x8xi32>
  %c1 = arith.constant 1 : i32
  affine.for %i = 0 to 8 {
    // CHECK: memref.alloc(){{.*}} : memref<1x8xi32>
    affine.for %j = 0 to 8 {
      affine.store %c1, %tmp[%i + 1, %j] : memref<8x8xi32>
    } {sym_name = "j"}
    // CHECK: affine.store %{{.*}}, %{{.*}}[0, %{{.*}}] : memref<1x8xi32>
    // CHECK: affine.load %{{.*}}[%{{.*}} - %{{.*}} - 1, %{{.*}}] : memref<1x8xi32>
  } {sym_name = "i"}
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    %alloc = transform.structured.match attributes {sym_name = "tmp"} in %root
      : (!transform.any_op) -> !transform.any_op
    %target = transform.allo.match_value 0 of %alloc kind 2
      : !transform.any_op -> !transform.any_value
    %axis = transform.structured.match attributes {sym_name = "i"} in %root
      : (!transform.any_op) -> !transform.any_op
    %local = transform.allo.buffer_at %target at %axis
      : (!transform.any_value, !transform.any_op) -> !transform.any_value
    transform.yield
  }
}

// -----

// CHECK-LABEL: func.func @buffer_at_affine_apply_offset
// CHECK: memref.alloc(){{.*}} : memref<1x8xi32>
// CHECK: affine.store %{{.*}}, %{{.*}}[0, %{{.*}}] : memref<1x8xi32>
// CHECK: affine.load %{{.*}}[%{{.*}} - %{{.*}} - 1, %{{.*}}] : memref<1x8xi32>
func.func @buffer_at_affine_apply_offset() {
  %tmp = memref.alloc() {sym_name = "tmp"} : memref<8x8xi32>
  %c1 = arith.constant 1 : i32
  affine.for %i = 0 to 8 {
    affine.for %j = 0 to 8 {
      %ip1 = affine.apply affine_map<(d0) -> (d0 + 1)>(%i)
      affine.store %c1, %tmp[%ip1, %j] : memref<8x8xi32>
    } {sym_name = "j"}
  } {sym_name = "i"}
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    %alloc = transform.structured.match attributes {sym_name = "tmp"} in %root
      : (!transform.any_op) -> !transform.any_op
    %target = transform.allo.match_value 0 of %alloc kind 2
      : !transform.any_op -> !transform.any_value
    %axis = transform.structured.match attributes {sym_name = "i"} in %root
      : (!transform.any_op) -> !transform.any_op
    %local = transform.allo.buffer_at %target at %axis
      : (!transform.any_value, !transform.any_op) -> !transform.any_value
    transform.yield
  }
}

// -----

// CHECK-LABEL: func.func @buffer_at_axis_tile
func.func @buffer_at_axis_tile() {
  %tmp = memref.alloc() {sym_name = "tmp"} : memref<64xi32>
  %c1 = arith.constant 1 : i32
  affine.for %i = 0 to 8 {
    // CHECK: memref.alloc(){{.*}} : memref<8xi32>
    affine.for %j = 0 to 8 {
      affine.store %c1, %tmp[8 * %i + %j] : memref<64xi32>
    } {sym_name = "j"}
    // CHECK: affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<8xi32>
    // CHECK: affine.load %{{.*}}[%{{.*}} - %{{.*}} * 8] : memref<8xi32>
  } {sym_name = "i"}
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    %alloc = transform.structured.match attributes {sym_name = "tmp"} in %root
      : (!transform.any_op) -> !transform.any_op
    %target = transform.allo.match_value 0 of %alloc kind 2
      : !transform.any_op -> !transform.any_value
    %axis = transform.structured.match attributes {sym_name = "i"} in %root
      : (!transform.any_op) -> !transform.any_op
    %local = transform.allo.buffer_at %target at %axis
      : (!transform.any_value, !transform.any_op) -> !transform.any_value
    transform.yield
  }
}

// -----

func.func @buffer_at_overlapping_axis_tiles_rejected() {
  %tmp = memref.alloc() {sym_name = "tmp"} : memref<16xi32>
  %c1 = arith.constant 1 : i32
  // expected-note @+1 {{different iterations of the selected axis access overlapping regions of the target buffer}}
  affine.for %i = 0 to 8 { // expected-error {{the target buffer cannot be made private to each iteration}}
    affine.for %j = 0 to 8 {
      affine.store %c1, %tmp[%i + %j] : memref<16xi32>
    } {sym_name = "j"}
  } {sym_name = "i"}
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    %alloc = transform.structured.match attributes {sym_name = "tmp"} in %root
      : (!transform.any_op) -> !transform.any_op
    %target = transform.allo.match_value 0 of %alloc kind 2
      : !transform.any_op -> !transform.any_value
    %axis = transform.structured.match attributes {sym_name = "i"} in %root
      : (!transform.any_op) -> !transform.any_op
    %local = transform.allo.buffer_at %target at %axis  // expected-error {{buffer_at failed}}
      : (!transform.any_value, !transform.any_op) -> !transform.any_value
    transform.yield
  }
}

// -----

// CHECK-LABEL: func.func @buffer_at_permuted_access
// CHECK: memref.alloc(){{.*}} : memref<8x1xi32>
// CHECK: affine.store %{{.*}}, %{{.*}}[%{{.*}}, 0] : memref<8x1xi32>
func.func @buffer_at_permuted_access() {
  %tmp = memref.alloc() {sym_name = "tmp"} : memref<8x8xi32>
  %c1 = arith.constant 1 : i32
  affine.for %i = 0 to 8 {
    affine.for %j = 0 to 8 {
      affine.store %c1, %tmp[%j, %i] : memref<8x8xi32>
    } {sym_name = "j"}
  } {sym_name = "i"}
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    %alloc = transform.structured.match attributes {sym_name = "tmp"} in %root
      : (!transform.any_op) -> !transform.any_op
    %target = transform.allo.match_value 0 of %alloc kind 2
      : !transform.any_op -> !transform.any_value
    %axis = transform.structured.match attributes {sym_name = "i"} in %root
      : (!transform.any_op) -> !transform.any_op
    %local = transform.allo.buffer_at %target at %axis
      : (!transform.any_value, !transform.any_op) -> !transform.any_value
    transform.yield
  }
}

// -----

func.func @buffer_at_multiple_access() {
  %tmp = memref.alloc() {sym_name = "tmp"} : memref<8x8xi32>
  affine.for %i = 0 to 8 {
    affine.for %j = 0 to 8 {
      %0 = affine.load %tmp[%i, %j] : memref<8x8xi32>
      affine.store %0, %tmp[%j, %i] : memref<8x8xi32>
    } {sym_name = "j"}
  } {sym_name = "i"}
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    %alloc = transform.structured.match attributes {sym_name = "tmp"} in %root
      : (!transform.any_op) -> !transform.any_op
    %target = transform.allo.match_value 0 of %alloc kind 2
      : !transform.any_op -> !transform.any_value
    %axis = transform.structured.match attributes {sym_name = "i"} in %root
      : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{buffer_at requires a bounded, realizable per-instance affine footprint for the target buffer}}
    %local = transform.allo.buffer_at %target at %axis
      : (!transform.any_value, !transform.any_op) -> !transform.any_value
    transform.yield
  }
}


// -----

func.func @buffer_at_memref_load_rejected() {
  %tmp = memref.alloc() {sym_name = "tmp_non_affine_load"} : memref<8x8xi32>
  affine.for %i = 0 to 8 { // expected-error {{only supports affine.load/store}}
    affine.for %j = 0 to 8 {
      %0 = memref.load %tmp[%i, %j] : memref<8x8xi32> // expected-note {{see memref.load op here}}
    } {sym_name = "j"}
  } {sym_name = "i"}
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    %alloc = transform.structured.match attributes {sym_name = "tmp_non_affine_load"} in %root
      : (!transform.any_op) -> !transform.any_op
    %target = transform.allo.match_value 0 of %alloc kind 2
      : !transform.any_op -> !transform.any_value
    %axis = transform.structured.match attributes {sym_name = "i"} in %root
      : (!transform.any_op) -> !transform.any_op
    %local = transform.allo.buffer_at %target at %axis  // expected-error {{buffer_at failed}}
      : (!transform.any_value, !transform.any_op) -> !transform.any_value
    transform.yield
  }
}

// -----

func.func @buffer_at_alias_view_rejected() {
  %tmp = memref.alloc() {sym_name = "tmp_alias"} : memref<8x8xi32>
  affine.for %i = 0 to 8 { // expected-error {{aliasing/view accesses}}
    affine.for %j = 0 to 8 {
      %view = memref.cast %tmp : memref<8x8xi32> to memref<?x?xi32> // expected-note {{see aliasing/view op here}}
    } {sym_name = "j"}
  } {sym_name = "i"}
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    %alloc = transform.structured.match attributes {sym_name = "tmp_alias"} in %root
      : (!transform.any_op) -> !transform.any_op
    %target = transform.allo.match_value 0 of %alloc kind 2
      : !transform.any_op -> !transform.any_value
    %axis = transform.structured.match attributes {sym_name = "i"} in %root
      : (!transform.any_op) -> !transform.any_op
    %local = transform.allo.buffer_at %target at %axis  // expected-error {{buffer_at failed}}
      : (!transform.any_value, !transform.any_op) -> !transform.any_value
    transform.yield
  }
}

// -----

// CHECK-LABEL: func.func @buffer_at_gemm_non_reduction
func.func @buffer_at_gemm_non_reduction(%a: memref<4x4xi32>,
                                        %b: memref<4x8xi32>) {
  %acc = memref.alloc() {sym_name = "acc"} : memref<4x8xi32>
  affine.for %i = 0 to 4 {
    // CHECK: memref.alloc(){{.*}} : memref<1x8xi32>
    affine.for %j = 0 to 8 {
      affine.for %r = 0 to 4 {
        %lhs = affine.load %a[%i, %r] : memref<4x4xi32>
        %rhs = affine.load %b[%r, %j] : memref<4x8xi32>
        %old = affine.load %acc[%i, %j] : memref<4x8xi32>
        %mul = arith.muli %lhs, %rhs : i32
        %new = arith.addi %old, %mul : i32
        affine.store %new, %acc[%i, %j] : memref<4x8xi32>
      } {sym_name = "r"}
    } {sym_name = "j"}
  } {sym_name = "i"}
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    %alloc = transform.structured.match attributes {sym_name = "acc"} in %root
      : (!transform.any_op) -> !transform.any_op
    %target = transform.allo.match_value 0 of %alloc kind 2
      : !transform.any_op -> !transform.any_value
    %axis = transform.structured.match attributes {sym_name = "i"} in %root
      : (!transform.any_op) -> !transform.any_op
   %local = transform.allo.buffer_at %target at %axis
      : (!transform.any_value, !transform.any_op) -> !transform.any_value
    transform.yield
  }
}

// -----

func.func @buffer_at_gemm_reduction_rejected(%a: memref<4x4xi32>,
                                             %b: memref<4x8xi32>) {
  %acc = memref.alloc() {sym_name = "acc"} : memref<4x8xi32>
  affine.for %i = 0 to 4 {
    // expected-note @+1 {{the target-buffer access pattern does not depend on the selected axis, so every iteration would use the same region}}
    affine.for %r = 0 to 4 { // expected-error {{the target buffer cannot be made private to each iteration}}
      affine.for %j = 0 to 8 {
        %lhs = affine.load %a[%i, %r] : memref<4x4xi32>
        %rhs = affine.load %b[%r, %j] : memref<4x8xi32>
        %old = affine.load %acc[%i, %j] : memref<4x8xi32>
        %mul = arith.muli %lhs, %rhs : i32
        %new = arith.addi %old, %mul : i32
        affine.store %new, %acc[%i, %j] : memref<4x8xi32>
      } {sym_name = "j"}
    } {sym_name = "r"}
  } {sym_name = "i"}
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    %alloc = transform.structured.match attributes {sym_name = "acc"} in %root
      : (!transform.any_op) -> !transform.any_op
    %target = transform.allo.match_value 0 of %alloc kind 2
      : !transform.any_op -> !transform.any_value
    %axis = transform.structured.match attributes {sym_name = "r"} in %root
      : (!transform.any_op) -> !transform.any_op
   %local = transform.allo.buffer_at %target at %axis  // expected-error {{buffer_at failed}}
      : (!transform.any_value, !transform.any_op) -> !transform.any_value
    transform.yield
  }
}

// -----

func.func @buffer_at_innermost_rejected() {
  %tmp = memref.alloc() {sym_name = "tmp"} : memref<8x8xi32>
  %c1 = arith.constant 1 : i32
  affine.for %i = 0 to 8 {
    affine.for %j = 0 to 8 {
      affine.store %c1, %tmp[%i, %j] : memref<8x8xi32>
    } {sym_name = "j"}
  } {sym_name = "i"}
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    %alloc = transform.structured.match attributes {sym_name = "tmp"} in %root
      : (!transform.any_op) -> !transform.any_op
    %target = transform.allo.match_value 0 of %alloc kind 2
      : !transform.any_op -> !transform.any_value
    %axis = transform.structured.match attributes {sym_name = "j"} in %root
      : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{cannot buffer at innermost loop axis}}
    %local = transform.allo.buffer_at %target at %axis
      : (!transform.any_value, !transform.any_op) -> !transform.any_value
    transform.yield
  }
}
