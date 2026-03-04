// RUN: allo-opt %s -split-input-file -transform-interpreter -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @partition_from_subview
func.func @partition_from_subview() {
  // CHECK: memref.alloc{{.*}} #part
  %local = memref.alloc() {sym_name = "local"} : memref<8xi32>
  // CHECK-NOT: memref.subview{{.*}} #part
  %slice = memref.subview %local[0] [4] [1] {sym_name = "slice"} : memref<8xi32> to memref<4xi32, strided<[1], offset: 0>>
  memref.dealloc %local : memref<8xi32>
  func.return
}

#partition = #allo.partition<[(0, Complete, 0), (1, Block, 2)]>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.op<"builtin.module">) {
    %view = transform.structured.match ops{["memref.subview"]} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"memref.subview">
    %slice = transform.allo.match_value 0 of %view
      : !transform.op<"memref.subview"> -> !transform.any_value
    transform.allo.partition %slice with #partition
      : !transform.any_value
    transform.yield
  }
}

// -----

// CHECK-LABEL: func.func @partition_func_arg
// CHECK: %arg0: memref<4xi32> {{.*}} #part
func.func @partition_func_arg(%arg0: memref<4xi32>) {
  func.return
}

#partition = #allo.partition<[(0, Complete, 0)]>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.op<"builtin.module">) {
    %kernel = transform.structured.match ops{["func.func"]} attributes {sym_name = "partition_func_arg"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"func.func">
    %arg = transform.allo.match_value 0 of %kernel
      : !transform.op<"func.func"> -> !transform.any_value
    transform.allo.partition %arg with #partition
      : !transform.any_value
    transform.yield
  }
}

// -----

// CHECK: (0,Complete,0), (1,Block,2)
// CHECK: func.func @partition_merge_dims
func.func @partition_merge_dims() {
  // CHECK: memref.alloc{{.*}} #part
  %buf = memref.alloc() {sym_name = "buf"} : memref<4x4xi32>
  memref.dealloc %buf : memref<4x4xi32>
  func.return
}

#block = #allo.partition<[(0, Complete, 0), (1, Block, 2)]>
#misc = #allo.partition<[(0, Complete, 0)]>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.op<"builtin.module">) {
    %alloc = transform.structured.match ops{["memref.alloc"]} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"memref.alloc">
    %buf = transform.allo.match_value 0 of %alloc
      : !transform.op<"memref.alloc"> -> !transform.any_value
    transform.allo.partition %buf with #block
      : !transform.any_value
    transform.allo.partition %buf with #misc
      : !transform.any_value
    transform.yield
  }
}

// -----

func.func @partition_invalid_existing_attr() {
  %local = memref.alloc() {sym_name = "local", allo.part = "bad"} : memref<8xi32>
  memref.dealloc %local : memref<8xi32>
  func.return
}

#part = #allo.partition<[(0, Complete, 0)]>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.op<"builtin.module">) {
    %alloc = transform.structured.match ops{["memref.alloc"]} attributes {sym_name = "local"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"memref.alloc">
    %local = transform.allo.match_value 0 of %alloc
      : !transform.op<"memref.alloc"> -> !transform.any_value
    // expected-error @below {{existing allo.part attribute is not a partition attribute}}
    transform.allo.partition %local with #part
      : !transform.any_value
    transform.yield
  }
}

// -----

func.func @partition_invalid_root_callee() -> memref<8xi32> {
  %local = memref.alloc() : memref<8xi32>
  func.return %local : memref<8xi32>
}

func.func @partition_invalid_root_not_alloc() {
  %buf = func.call @partition_invalid_root_callee() {sym_name = "callbuf"} : () -> memref<8xi32>
  memref.dealloc %buf : memref<8xi32>
  func.return
}

#part = #allo.partition<[(0, Complete, 0)]>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.op<"builtin.module">) {
    %call = transform.structured.match ops{["func.call"]} attributes {sym_name = "callbuf"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"func.call">
    %v = transform.allo.match_value 0 of %call
      : !transform.op<"func.call"> -> !transform.any_value
    // expected-error @below {{partition target root must resolve to memref.alloc, memref.alloca, memref.get_global, or function argument}}
    transform.allo.partition %v with #part
      : !transform.any_value
    transform.yield
  }
}
