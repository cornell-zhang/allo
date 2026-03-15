// RUN: allo-opt %s -split-input-file -transform-interpreter -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @reuse_at_basic
// CHECK: %[[SRC:.*]] = memref.alloc() {sym_name = "in_buf"} : memref<8x8xi32>
// CHECK: %[[REUSE:.*]] = memref.alloc() {sym_name = "in_buf::reuse"} : memref<3xi32>
// CHECK-NOT: reuse_head
// CHECK: affine.for %{{.*}} = 0 to 8 {
// CHECK:   %{{.*}} = affine.for %{{.*}} = 0 to 6 iter_args(%[[HEAD:.*]] = %c0) -> (index) {
// CHECK:     %{{.*}} = arith.addi %[[HEAD]], %c1 : index
// CHECK:     %{{.*}} = arith.remui %{{.*}}, %c3 : index
// CHECK:     affine.if #set(
// CHECK:       %{{.*}} = affine.for %{{.*}} = 0 to 1 iter_args(%{{.*}} = %{{.*}}) -> (index) {
// CHECK:       memref.store %{{.*}}, %[[REUSE]][%{{.*}}] : memref<3xi32>
// CHECK:     affine.if #set1(
// CHECK:       %{{.*}} = affine.load %[[SRC]][%{{.*}}, %{{.*}}] : memref<8x8xi32>
// CHECK:       affine.store %{{.*}}, %[[REUSE]][0] : memref<3xi32>
// CHECK:     } else {
// CHECK:       %{{.*}} = memref.load %[[REUSE]][%{{.*}}] : memref<3xi32>
// CHECK:       %{{.*}} = memref.load %[[REUSE]][%{{.*}}] : memref<3xi32>
// CHECK:       %{{.*}} = memref.load %[[REUSE]][%{{.*}}] : memref<3xi32>
// CHECK:     affine.yield %{{.*}} : index
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
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.op<"builtin.module">) {
    %alloc = transform.structured.match ops{["memref.alloc"]} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"memref.alloc">
    %target = transform.allo.match_value 0 of %alloc kind 2
      : !transform.op<"memref.alloc"> -> !transform.any_value
    %axis = transform.structured.match attributes {sym_name = "x"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"affine.for">
    %reuse = transform.allo.reuse_at %target at %axis ring
      : (!transform.any_value, !transform.op<"affine.for">) -> !transform.any_value
    transform.yield
  }
}

// -----

// CHECK-LABEL: func.func @reuse_at_basic_no_ring
// CHECK: %[[SRC:.*]] = memref.alloc() {sym_name = "plain_buf"} : memref<8x8xi32>
// CHECK: %[[REUSE:.*]] = memref.alloc() {sym_name = "plain_buf::reuse"} : memref<3xi32>
// CHECK-NOT: reuse_head
// CHECK: affine.for %{{.*}} = 0 to 8 {
// CHECK:   affine.for %{{.*}} = 0 to 6 {
// CHECK:     affine.if
// CHECK:       affine.for %{{.*}} = 0 to 2 {
// CHECK:         %{{.*}} = affine.load %[[REUSE]][%{{.*}}] : memref<3xi32>
// CHECK:         affine.store %{{.*}}, %[[REUSE]][%{{.*}}] : memref<3xi32>
// CHECK:       affine.for %{{.*}} = 0 to 1 {
// CHECK:         affine.store %{{.*}}, %[[REUSE]][%{{.*}}] : memref<3xi32>
// CHECK-NOT: = affine.if #set1
// CHECK:     affine.if #set1(
// CHECK:       %{{.*}} = affine.load %[[SRC]][%{{.*}}, %{{.*}}] : memref<8x8xi32>
// CHECK:       affine.store %{{.*}}, %[[REUSE]][0] : memref<3xi32>
// CHECK:       %{{.*}} = affine.load %[[SRC]][%{{.*}}, %{{.*}} + 1] : memref<8x8xi32>
// CHECK:       affine.store %{{.*}}, %[[REUSE]][1] : memref<3xi32>
// CHECK:       %{{.*}} = affine.load %[[SRC]][%{{.*}}, %{{.*}} + 2] : memref<8x8xi32>
// CHECK:       affine.store %{{.*}}, %[[REUSE]][2] : memref<3xi32>
// CHECK:       %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i32
// CHECK:       %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i32
// CHECK:       affine.store %{{.*}}, %arg0[%{{.*}}, %{{.*}}] : memref<8x6xi32>
// CHECK:     } else {
// CHECK:       %{{.*}} = affine.load %[[REUSE]][0] : memref<3xi32>
// CHECK:       %{{.*}} = affine.load %[[REUSE]][1] : memref<3xi32>
// CHECK:       %{{.*}} = affine.load %[[REUSE]][2] : memref<3xi32>
// CHECK:       %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i32
// CHECK:       %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i32
// CHECK:       affine.store %{{.*}}, %arg0[%{{.*}}, %{{.*}}] : memref<8x6xi32>
func.func @reuse_at_basic_no_ring(%out: memref<8x6xi32>) {
  %plain_buf = memref.alloc() {sym_name = "plain_buf"} : memref<8x8xi32>
  affine.for %y = 0 to 8 {
    affine.for %x = 0 to 6 {
      %a0 = affine.load %plain_buf[%y, %x] : memref<8x8xi32>
      %a1 = affine.load %plain_buf[%y, %x + 1] : memref<8x8xi32>
      %a2 = affine.load %plain_buf[%y, %x + 2] : memref<8x8xi32>
      %t0 = arith.addi %a0, %a1 : i32
      %t1 = arith.addi %t0, %a2 : i32
      affine.store %t1, %out[%y, %x] : memref<8x6xi32>
    } {sym_name = "x"}
  } {sym_name = "y"}
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.op<"builtin.module">) {
    %alloc = transform.structured.match ops{["memref.alloc"]} attributes {sym_name = "plain_buf"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"memref.alloc">
    %target = transform.allo.match_value 0 of %alloc kind 2
      : !transform.op<"memref.alloc"> -> !transform.any_value
    %axis = transform.structured.match attributes {sym_name = "x"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"affine.for">
    %reuse = transform.allo.reuse_at %target at %axis
      : (!transform.any_value, !transform.op<"affine.for">) -> !transform.any_value
    transform.yield
  }
}

// -----

// CHECK-LABEL: func.func @reuse_at_avgpool_nchw_like
// CHECK: %[[REUSE:.*]] = memref.alloc() {sym_name = "avgpool_buf::reuse"} : memref<3x6xf32>
// CHECK-NOT: reuse_head
// CHECK: affine.for %{{.*}} = 0 to 2 {
// CHECK:   affine.for %{{.*}} = 0 to 2 {
// CHECK:     %{{.*}} = affine.for %{{.*}} = 0 to 3 iter_args(%{{.*}} = %c0) -> (index) {
// CHECK:       affine.if #set(
// CHECK:         %{{.*}} = affine.for %{{.*}} = 0 to 2 iter_args(%{{.*}} = %{{.*}}) -> (index) {
// CHECK:           affine.for %{{.*}} = 0 to 6 {
// CHECK:             memref.store %{{.*}}, %[[REUSE]][%{{.*}}, %{{.*}}] : memref<3x6xf32>
// CHECK: affine.for %{{.*}} = 0 to 5 {
// CHECK:   affine.for %{{.*}} = 0 to 3 {
// CHECK:     affine.for %{{.*}} = 0 to 2 {
// CHECK:       affine.if #set1(
// CHECK:       } else {
// CHECK:         %{{.*}} = memref.load %[[REUSE]][%{{.*}}, %{{.*}}] : memref<3x6xf32>
func.func @reuse_at_avgpool_nchw_like(%out: memref<2x2x3x5xf32>) {
  %avgpool_buf = memref.alloc() {sym_name = "avgpool_buf"} : memref<2x2x8x10xf32>
  affine.for %n = 0 to 2 {
    affine.for %c = 0 to 2 {
      affine.for %h = 0 to 3 {
        affine.for %w = 0 to 5 {
          affine.for %rh = 0 to 3 {
            affine.for %rw = 0 to 2 {
              %v = affine.load %avgpool_buf[%n, %c, %h * 2 + %rh, %w + %rw] : memref<2x2x8x10xf32>
              affine.store %v, %out[%n, %c, %h, %w] : memref<2x2x3x5xf32>
            } {sym_name = "rw"}
          } {sym_name = "rh"}
        } {sym_name = "w"}
      } {sym_name = "h"}
    } {sym_name = "c"}
  } {sym_name = "n"}
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.op<"builtin.module">) {
    %alloc = transform.structured.match ops{["memref.alloc"]} attributes {sym_name = "avgpool_buf"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"memref.alloc">
    %target = transform.allo.match_value 0 of %alloc kind 2
      : !transform.op<"memref.alloc"> -> !transform.any_value
    %axis = transform.structured.match attributes {sym_name = "h"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"affine.for">
    %reuse = transform.allo.reuse_at %target at %axis ring
      : (!transform.any_value, !transform.op<"affine.for">) -> !transform.any_value
    transform.yield
  }
}

// -----

// CHECK-LABEL: func.func @reuse_at_chained_no_ring_then_ring
// CHECK: %[[LINE:.*]] = memref.alloc() {sym_name = "chain_tail_ring_buf::reuse"} : memref<3x8xi32>
// CHECK: %[[WINDOW:.*]] = memref.alloc() {sym_name = "chain_tail_ring_buf::reuse::reuse"} : memref<3x3xi32>
// CHECK-NOT: reuse_head
// CHECK: affine.for %{{.*}} = 0 to 6 {
// CHECK:   %{{.*}} = affine.for %{{.*}} = 0 to 6 iter_args(%{{.*}} = %c0) -> (index) {
// CHECK:     affine.if #set(
// CHECK:       %{{.*}} = affine.for %{{.*}} = 0 to 1 iter_args(%{{.*}} = %{{.*}}) -> (index) {
// CHECK:     %{{.*}} = affine.load %[[LINE]][%{{.*}}, %{{.*}}] : memref<3x8xi32>
// CHECK:     memref.store %{{.*}}, %[[WINDOW]][%{{.*}}, %{{.*}}] : memref<3x3xi32>
// CHECK:     %{{.*}} = memref.load %[[WINDOW]][%{{.*}}, %{{.*}}] : memref<3x3xi32>
func.func @reuse_at_chained_no_ring_then_ring(%out: memref<6x6xi32>) {
  %chain_tail_ring_buf = memref.alloc() {sym_name = "chain_tail_ring_buf"} : memref<8x8xi32>
  affine.for %y = 0 to 6 {
    affine.for %x = 0 to 6 {
      affine.for %ry = 0 to 3 {
        affine.for %rx = 0 to 3 {
          %v = affine.load %chain_tail_ring_buf[%y + %ry, %x + %rx] : memref<8x8xi32>
          affine.store %v, %out[%y, %x] : memref<6x6xi32>
        } {sym_name = "rx"}
      } {sym_name = "ry"}
    } {sym_name = "x"}
  } {sym_name = "y"}
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.op<"builtin.module">) {
    %alloc = transform.structured.match ops{["memref.alloc"]} attributes {sym_name = "chain_tail_ring_buf"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"memref.alloc">
    %target = transform.allo.match_value 0 of %alloc kind 2
      : !transform.op<"memref.alloc"> -> !transform.any_value
    %y = transform.structured.match attributes {sym_name = "y"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"affine.for">
    %line = transform.allo.reuse_at %target at %y
      : (!transform.any_value, !transform.op<"affine.for">) -> !transform.any_value
    %x = transform.structured.match attributes {sym_name = "x"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"affine.for">
    %window = transform.allo.reuse_at %line at %x ring
      : (!transform.any_value, !transform.op<"affine.for">) -> !transform.any_value
    transform.yield
  }
}

// -----

// CHECK-LABEL: func.func @reuse_at_stencil
// CHECK: %[[REUSE:.*]] = memref.alloc() : memref<3x16xi32>
// CHECK-NOT: reuse_head
// CHECK: affine.for %{{.*}} = 0 to 10 {
// CHECK:   %{{.*}} = affine.for %{{.*}} = 1 to 15 iter_args(%{{.*}} = %c0) -> (index) {
// CHECK:     %{{.*}} = affine.apply
// CHECK:     affine.if #set(
// CHECK:       %{{.*}} = affine.for %{{.*}} = 0 to 1 iter_args(%{{.*}} = %{{.*}}) -> (index) {
// CHECK:         affine.for %{{.*}} = 0 to 16 {
// CHECK:           memref.store %{{.*}}, %[[REUSE]][%{{.*}}, %{{.*}}] : memref<3x16xi32>
// CHECK:     affine.for %{{.*}} = 1 to 15 {
// CHECK:       affine.if #set1(
// CHECK:         affine.store %{{.*}}, %[[REUSE]][0, %{{.*}}] : memref<3x16xi32>
// CHECK:         affine.store %{{.*}}, %[[REUSE]][2, %{{.*}}] : memref<3x16xi32>
// CHECK:       } else {
// CHECK:         %{{.*}} = memref.load %[[REUSE]][%{{.*}}, %{{.*}}] : memref<3x16xi32>
func.func @reuse_at_stencil(%A: memref<16x16xi32>, %out: memref<16x16xi32>) {
  %c5 = arith.constant 5 : i32
  affine.for %t = 0 to 10 {
    affine.for %x = 1 to 15 {
      affine.for %y = 1 to 15 {
        %0 = affine.load %A[%x - 1, %y] : memref<16x16xi32>
        %1 = affine.load %A[%x + 1, %y] : memref<16x16xi32>
        %2 = affine.load %A[%x, %y + 1] : memref<16x16xi32>
        %3 = affine.load %A[%x, %y - 1] : memref<16x16xi32>
        %4 = affine.load %A[%x, %y] : memref<16x16xi32>
        %5 = arith.addi %0, %1 : i32
        %6 = arith.addi %5, %2 : i32
        %7 = arith.addi %6, %3 : i32
        %8 = arith.addi %7, %4 : i32
        %9 = arith.divsi %8, %c5 : i32
        affine.store %9, %out[%x, %y] : memref<16x16xi32>
      } {sym_name = "y"}
    } {sym_name = "x"}
  } {sym_name = "t"}
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.op<"builtin.module">) {
    %f = transform.structured.match ops{["func.func"]} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"func.func">
    %target = transform.allo.match_value 0 of %f kind 0
      : !transform.op<"func.func"> -> !transform.any_value
    %axis = transform.structured.match attributes {sym_name = "x"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"affine.for">
    %reuse = transform.allo.reuse_at %target at %axis ring
      : (!transform.any_value, !transform.op<"affine.for">) -> !transform.any_value
    transform.yield
  }
}

// -----

// CHECK-LABEL: func.func @reuse_at_lattice_window_direct
// CHECK: %[[REUSE:.*]] = memref.alloc() {sym_name = "lattice_direct_buf::reuse"} : memref<3xi32>
// CHECK-NOT: reuse_head
// CHECK: affine.for %{{.*}} = 0 to 8 {
// CHECK:   %{{.*}} = affine.for %{{.*}} = 0 to 8 step 2 iter_args(%{{.*}} = %c0) -> (index) {
// CHECK:     affine.if #set(
// CHECK:       %[[FIRST_SLOT:.*]] = arith.remui %{{.*}}, %c3 : index
// CHECK:       affine.for %{{.*}} = 0 to 1 iter_args(%[[SLOT:.*]] = %[[FIRST_SLOT]]) -> (index) {
// CHECK:         %{{.*}} = affine.load %{{.*}}[%{{.*}}, %{{.*}} * 2 + (%{{.*}} floordiv 2) * 2 + 4] : memref<8x16xi32>
// CHECK:         memref.store %{{.*}}, %[[REUSE]][%[[SLOT]]] : memref<3xi32>
// CHECK:     affine.if #set1(
// CHECK:       %{{.*}} = affine.load %{{.*}}[%{{.*}}, %{{.*}} + %{{.*}} * 2] : memref<8x16xi32>
// CHECK:     } else {
// CHECK:       %{{.*}} = memref.load %[[REUSE]][%{{.*}}] : memref<3xi32>
func.func @reuse_at_lattice_window_direct(%out: memref<8x4xi32>) {
  %lattice_direct_buf = memref.alloc() {sym_name = "lattice_direct_buf"} : memref<8x16xi32>
  affine.for %y = 0 to 8 {
    affine.for %x = 0 to 8 step 2 {
      affine.for %r = 0 to 3 {
        %v = affine.load %lattice_direct_buf[%y, %x + %r * 2] : memref<8x16xi32>
        affine.store %v, %out[%y, %x floordiv 2] : memref<8x4xi32>
      } {sym_name = "r"}
    } {sym_name = "x"}
  } {sym_name = "y"}
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.op<"builtin.module">) {
    %alloc = transform.structured.match ops{["memref.alloc"]} attributes {sym_name = "lattice_direct_buf"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"memref.alloc">
    %target = transform.allo.match_value 0 of %alloc kind 2
      : !transform.op<"memref.alloc"> -> !transform.any_value
    %axis = transform.structured.match attributes {sym_name = "x"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"affine.for">
    %reuse = transform.allo.reuse_at %target at %axis ring
      : (!transform.any_value, !transform.op<"affine.for">) -> !transform.any_value
    transform.yield
  }
}

// -----

// CHECK-LABEL: func.func @reuse_at_forward_window
// CHECK: %[[REUSE:.*]] = memref.alloc() : memref<2xi32>
// CHECK-NOT: reuse_head
// CHECK: affine.for %{{.*}} = 0 to 16 {
// CHECK:   affine.for %{{.*}} = 0 to 16 {
// CHECK:     affine.if #set(
// CHECK:       affine.for %{{.*}} = 0 to 1 {
// CHECK:         %{{.*}} = affine.load %[[REUSE]][%{{.*}} + 1] : memref<2xi32>
// CHECK:     affine.if #set1(
// CHECK:       %{{.*}} = affine.load %{{.*}}[%{{.*}}, %{{.*}} + 2] : memref<16x20xi32>
// CHECK:       %{{.*}} = affine.load %{{.*}}[%{{.*}}, %{{.*}} + 3] : memref<16x20xi32>
// CHECK:     } else {
// CHECK:       %{{.*}} = affine.load %[[REUSE]][0] : memref<2xi32>
// CHECK:       %{{.*}} = affine.load %[[REUSE]][1] : memref<2xi32>
func.func @reuse_at_forward_window(%A: memref<16x20xi32>, %out: memref<16x16xi32>) {
  affine.for %x = 0 to 16 {
    affine.for %y = 0 to 16 {
      %0 = affine.load %A[%x, %y + 2] : memref<16x20xi32>
      %1 = affine.load %A[%x, %y + 3] : memref<16x20xi32>
      %2 = arith.addi %0, %1 : i32
      affine.store %2, %out[%x, %y] : memref<16x16xi32>
    } {sym_name = "y"}
  } {sym_name = "x"}
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.op<"builtin.module">) {
    %f = transform.structured.match ops{["func.func"]} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"func.func">
    %target = transform.allo.match_value 0 of %f kind 0
      : !transform.op<"func.func"> -> !transform.any_value
    %axis = transform.structured.match attributes {sym_name = "y"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"affine.for">
    %reuse = transform.allo.reuse_at %target at %axis
      : (!transform.any_value, !transform.op<"affine.for">) -> !transform.any_value
    transform.yield
  }
}

// -----

// CHECK: #set = affine_set<(d0) : (d0 - 1 >= 0, -d0 + 2 >= 0)>
// CHECK-LABEL: func.func @reuse_at_tiled_inner_tail_boundary
// CHECK: %[[REUSE:.*]] = memref.alloc() {sym_name = "tiled_tail_buf::reuse"} : memref<3x3xi32>
// CHECK-NOT: reuse_head
// CHECK: affine.for %{{.*}} = 0 to 2 {
// CHECK:   affine.for %{{.*}} = 0 to 8 {
// CHECK:     %{{.*}} = affine.for %{{.*}} = 0 to 4 iter_args(%[[HEAD:.*]] = %c0) -> (index) {
// CHECK:       %{{.*}} = arith.cmpi sge, %{{.*}}, %c1 : index
// CHECK:       %{{.*}} = arith.cmpi sle, %{{.*}}, %c2 : index
// CHECK:       %[[ACTIVE:.*]] = arith.andi %{{.*}}, %{{.*}} : i1
// CHECK:       affine.if #set(
// CHECK:       %{{.*}} = arith.select %[[ACTIVE]], %{{.*}}, %[[HEAD]] : index
func.func @reuse_at_tiled_inner_tail_boundary(%out: memref<8x2x4xi32>) {
  %tiled_tail_buf = memref.alloc() {sym_name = "tiled_tail_buf"} : memref<10x10xi32>
  affine.for %xo = 0 to 2 {
    affine.for %y = 0 to 8 {
      affine.for %xi = 0 to 4 {
        affine.for %r = 0 to 3 {
          affine.for %c = 0 to 3 {
            %v = affine.load %tiled_tail_buf[%y + %r, %xo * 4 + %xi + %c] : memref<10x10xi32>
            affine.store %v, %out[%y, %xo, %xi] : memref<8x2x4xi32>
          } {sym_name = "c"}
        } {sym_name = "r"}
      } {sym_name = "xi"}
    } {sym_name = "y"}
  } {sym_name = "xo"}
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.op<"builtin.module">) {
    %alloc = transform.structured.match ops{["memref.alloc"]} attributes {sym_name = "tiled_tail_buf"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"memref.alloc">
    %target = transform.allo.match_value 0 of %alloc kind 2
      : !transform.op<"memref.alloc"> -> !transform.any_value
    %axis = transform.structured.match attributes {sym_name = "xi"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"affine.for">
    %reuse = transform.allo.reuse_at %target at %axis ring
      : (!transform.any_value, !transform.op<"affine.for">) -> !transform.any_value
    transform.yield
  }
}

// -----

// CHECK-LABEL: func.func @reuse_at_chained_stencil_yx
// CHECK: %[[LINE:.*]] = memref.alloc() : memref<3x8xi32>
// CHECK: %[[WINDOW:.*]] = memref.alloc() : memref<3x3xi32>
// CHECK-NOT: reuse_head
// CHECK: %{{.*}} = affine.load %[[LINE]][%{{.*}}, %{{.*}}] : memref<3x8xi32>
// CHECK: affine.store %{{.*}}, %[[WINDOW]][%{{.*}}, %{{.*}}] : memref<3x3xi32>
func.func @reuse_at_chained_stencil_yx(%A: memref<10x10xi32>, %out: memref<6x6xi32>) {
  affine.for %y = 1 to 7 {
    affine.for %x = 1 to 7 {
      affine.for %ry = 0 to 3 {
        affine.for %rx = 0 to 3 {
          %v = affine.load %A[%y + %ry - 1, %x + %rx - 1] : memref<10x10xi32>
          affine.store %v, %out[%y - 1, %x - 1] : memref<6x6xi32>
        } {sym_name = "rx"}
      } {sym_name = "ry"}
    } {sym_name = "x"}
  } {sym_name = "y"}
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.op<"builtin.module">) {
    %f = transform.structured.match ops{["func.func"]} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"func.func">
    %target = transform.allo.match_value 0 of %f kind 0
      : !transform.op<"func.func"> -> !transform.any_value
    %y = transform.structured.match attributes {sym_name = "y"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"affine.for">
    %line = transform.allo.reuse_at %target at %y
      : (!transform.any_value, !transform.op<"affine.for">) -> !transform.any_value
    %x = transform.structured.match attributes {sym_name = "x"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"affine.for">
    %window = transform.allo.reuse_at %line at %x
      : (!transform.any_value, !transform.op<"affine.for">) -> !transform.any_value
    transform.yield
  }
}

// -----

#fake_set = affine_set<(d0) : (-d0 >= 0)>

func.func @reuse_at_extension_only_wrapper_rejected(%A: memref<8x8xi32>, %out: memref<6x6xi32>) {
  %line = memref.alloc() {sym_name = "fake_chain_line"} : memref<3x8xi32>
  affine.for %y = 0 to 6 {
    affine.for %x = 0 to 6 {
      affine.for %ry = 0 to 3 {
        affine.for %rx = 0 to 3 {
          %v = affine.if #fake_set(%y) -> i32 {
            %s = affine.load %A[%y + %ry, %x + %rx] : memref<8x8xi32>
            affine.store %s, %line[%ry, %x + %rx] : memref<3x8xi32>
            affine.yield %s : i32
          } else {
            %idx = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%x, %rx)
            %s = memref.load %line[%ry, %idx] : memref<3x8xi32>
            affine.yield %s : i32
          }
          affine.store %v, %out[%y, %x] : memref<6x6xi32>
        } {sym_name = "rx"}
      } {sym_name = "ry"}
    } {sym_name = "x"}
  } {sym_name = "y"}
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.op<"builtin.module">) {
    %alloc = transform.structured.match ops{["memref.alloc"]} attributes {sym_name = "fake_chain_line"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"memref.alloc">
    %target = transform.allo.match_value 0 of %alloc kind 2
      : !transform.op<"memref.alloc"> -> !transform.any_value
    %axis = transform.structured.match attributes {sym_name = "x"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"affine.for">
    // expected-error @below {{classify loop roles}}
    %reuse = transform.allo.reuse_at %target at %axis
      : (!transform.any_value, !transform.op<"affine.for">) -> !transform.any_value
    transform.yield
  }
}

// -----

// CHECK-LABEL: func.func @reuse_at_chained_prefix_step2
// CHECK: %[[LINE:.*]] = memref.alloc() : memref<3x3xi32>
// CHECK: %[[WINDOW:.*]] = memref.alloc() : memref<7x3xi32>
// CHECK-NOT: reuse_head
// CHECK: %{{.*}} = affine.load %[[LINE]][%{{.*}}, %{{.*}} * 2 + %{{.*}} + 1] : memref<3x3xi32>
// CHECK: affine.store %{{.*}}, %[[WINDOW]][%{{.*}}, %{{.*}} + 1] : memref<7x3xi32>
// CHECK: %{{.*}} = affine.load %[[WINDOW]][%{{.*}}, %{{.*}}] : memref<7x3xi32>
func.func @reuse_at_chained_prefix_step2(%A: memref<8x20xi32>, %out: memref<5x6xi32>) {
  affine.for %x = 0 to 6 {
    affine.for %y = 0 to 5 {
      affine.for %ry = 0 to 3 {
        affine.for %rx = 0 to 3 {
          %v = affine.load %A[%y + %ry, %x * 2 + %rx] : memref<8x20xi32>
          affine.store %v, %out[%y, %x] : memref<5x6xi32>
        } {sym_name = "rx"}
      } {sym_name = "ry"}
    } {sym_name = "y"}
  } {sym_name = "x"}
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.op<"builtin.module">) {
    %f = transform.structured.match ops{["func.func"]} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"func.func">
    %target = transform.allo.match_value 0 of %f kind 0
      : !transform.op<"func.func"> -> !transform.any_value
    %y = transform.structured.match attributes {sym_name = "y"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"affine.for">
    %line = transform.allo.reuse_at %target at %y
      : (!transform.any_value, !transform.op<"affine.for">) -> !transform.any_value
    %x = transform.structured.match attributes {sym_name = "x"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"affine.for">
    %window = transform.allo.reuse_at %line at %x
      : (!transform.any_value, !transform.op<"affine.for">) -> !transform.any_value
    transform.yield
  }
}

// -----

func.func @reuse_at_chained_ring_middle_rejected(%out: memref<6x6xi32>) {
  %chain_middle_ring_buf = memref.alloc() {sym_name = "chain_middle_ring_buf"} : memref<8x8xi32>
  affine.for %y = 0 to 6 {
    affine.for %x = 0 to 6 {
      affine.for %ry = 0 to 3 {
        affine.for %rx = 0 to 3 {
          %v = affine.load %chain_middle_ring_buf[%y + %ry, %x + %rx] : memref<8x8xi32>
          affine.store %v, %out[%y, %x] : memref<6x6xi32>
        } {sym_name = "rx"}
      } {sym_name = "ry"}
    } {sym_name = "x"}
  } {sym_name = "y"}
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.op<"builtin.module">) {
    %alloc = transform.structured.match ops{["memref.alloc"]} attributes {sym_name = "chain_middle_ring_buf"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"memref.alloc">
    %target = transform.allo.match_value 0 of %alloc kind 2
      : !transform.op<"memref.alloc"> -> !transform.any_value
    %y = transform.structured.match attributes {sym_name = "y"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"affine.for">
    %line = transform.allo.reuse_at %target at %y ring
      : (!transform.any_value, !transform.op<"affine.for">) -> !transform.any_value
    %x = transform.structured.match attributes {sym_name = "x"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"affine.for">
    // expected-error @below {{classify loop roles}}
    %window = transform.allo.reuse_at %line at %x
      : (!transform.any_value, !transform.op<"affine.for">) -> !transform.any_value
    transform.yield
  }
}

// -----

func.func @reuse_at_noncontiguous_window_rejected(%out: memref<8x6xi32>) {
  %in_buf = memref.alloc() {sym_name = "noncontig_buf"} : memref<8x16xi32>
  affine.for %y = 0 to 8 {
    affine.for %x = 0 to 6 {
      affine.for %r = 0 to 3 {
        // expected-note @+1 {{dense slot-space lattice}}
        %v = affine.load %in_buf[%y, %x + %r * 2] : memref<8x16xi32> // expected-error {{bounded strided affine-lattice footprints}}
        affine.store %v, %out[%y, %x] : memref<8x6xi32>
      } {sym_name = "r"}
    } {sym_name = "x"}
  } {sym_name = "y"}
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.op<"builtin.module">) {
    %alloc = transform.structured.match ops{["memref.alloc"]} attributes {sym_name = "noncontig_buf"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"memref.alloc">
    %target = transform.allo.match_value 0 of %alloc kind 2
      : !transform.op<"memref.alloc"> -> !transform.any_value
    %axis = transform.structured.match attributes {sym_name = "x"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"affine.for">
    // expected-error @below {{analyze reuse candidate accesses}}
    %reuse = transform.allo.reuse_at %target at %axis
      : (!transform.any_value, !transform.op<"affine.for">) -> !transform.any_value
    transform.yield
  }
}

// -----

func.func @reuse_at_incompatible_lattice_stride_rejected(%out: memref<8x2xi32>) {
  %in_buf = memref.alloc() {sym_name = "bad_lattice_buf"} : memref<8x24xi32>
  affine.for %y = 0 to 8 {
    affine.for %x = 0 to 12 step 6 {
      affine.for %r = 0 to 3 {
        // expected-note @+1 {{dense slot-space lattice}}
        %a = affine.load %in_buf[%y, %x + %r * 2] : memref<8x24xi32> // expected-error {{bounded strided affine-lattice footprints}}
        %b = affine.load %in_buf[%y, %x + %r * 3] : memref<8x24xi32>
        %c = arith.addi %a, %b : i32
        affine.store %c, %out[%y, %x floordiv 6] : memref<8x2xi32>
      } {sym_name = "r"}
    } {sym_name = "x"}
  } {sym_name = "y"}
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.op<"builtin.module">) {
    %alloc = transform.structured.match ops{["memref.alloc"]} attributes {sym_name = "bad_lattice_buf"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"memref.alloc">
    %target = transform.allo.match_value 0 of %alloc kind 2
      : !transform.op<"memref.alloc"> -> !transform.any_value
    %axis = transform.structured.match attributes {sym_name = "x"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"affine.for">
    // expected-error @below {{analyze reuse candidate accesses}}
    %reuse = transform.allo.reuse_at %target at %axis
      : (!transform.any_value, !transform.op<"affine.for">) -> !transform.any_value
    transform.yield
  }
}

// -----

func.func @reuse_at_stride_no_overlap_rejected(%out: memref<8x4xi32>) {
  %in_buf = memref.alloc() {sym_name = "no_overlap_buf"} : memref<8x16xi32>
  affine.for %y = 0 to 8 {
    // expected-note @+1 {{sliding step}}
    affine.for %x = 0 to 4 { // expected-error {{cross-iteration overlap}}
      affine.for %r = 0 to 3 {
        %v = affine.load %in_buf[%y, %x * 3 + %r] : memref<8x16xi32>
        affine.store %v, %out[%y, %x] : memref<8x4xi32>
      } {sym_name = "r"}
    } {sym_name = "x"}
  } {sym_name = "y"}
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.op<"builtin.module">) {
    %alloc = transform.structured.match ops{["memref.alloc"]} attributes {sym_name = "no_overlap_buf"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"memref.alloc">
    %target = transform.allo.match_value 0 of %alloc kind 2
      : !transform.op<"memref.alloc"> -> !transform.any_value
    %axis = transform.structured.match attributes {sym_name = "x"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"affine.for">
    // expected-error @below {{analyze reuse candidate accesses}}
    %reuse = transform.allo.reuse_at %target at %axis
      : (!transform.any_value, !transform.op<"affine.for">) -> !transform.any_value
    transform.yield
  }
}

// -----

func.func @reuse_at_target_write_hazard(%out: memref<8x6xi32>) {
  %in_buf = memref.alloc() {sym_name = "hazard_buf"} : memref<8x8xi32>
  %c0 = arith.constant 0 : i32
  affine.for %y = 0 to 8 {
    affine.for %x = 0 to 6 { // expected-error {{read-only}}
      %a0 = affine.load %in_buf[%y, %x] : memref<8x8xi32>
      %a1 = affine.load %in_buf[%y, %x + 1] : memref<8x8xi32>
      // expected-note @+1 {{write op}}
      affine.store %c0, %in_buf[%y, %x + 2] : memref<8x8xi32>
      %t0 = arith.addi %a0, %a1 : i32
      affine.store %t0, %out[%y, %x] : memref<8x6xi32>
    } {sym_name = "x"}
  } {sym_name = "y"}
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.op<"builtin.module">) {
    %alloc = transform.structured.match ops{["memref.alloc"]} attributes {sym_name = "hazard_buf"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"memref.alloc">
    %target = transform.allo.match_value 0 of %alloc kind 2
      : !transform.op<"memref.alloc"> -> !transform.any_value
    %axis = transform.structured.match attributes {sym_name = "x"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"affine.for">
    // expected-error @below {{analyze reuse candidate accesses}}
    %reuse = transform.allo.reuse_at %target at %axis
      : (!transform.any_value, !transform.op<"affine.for">) -> !transform.any_value
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
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.op<"builtin.module">) {
    %alloc = transform.structured.match ops{["memref.alloc"]} attributes {sym_name = "classify_buf"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"memref.alloc">
    %target = transform.allo.match_value 0 of %alloc kind 2
      : !transform.op<"memref.alloc"> -> !transform.any_value
    %axis = transform.structured.match attributes {sym_name = "r"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"affine.for">
    // expected-error @below {{reduction loop}}
    %reuse = transform.allo.reuse_at %target at %axis
      : (!transform.any_value, !transform.op<"affine.for">) -> !transform.any_value
    transform.yield
  }
}
