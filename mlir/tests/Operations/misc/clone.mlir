// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: allo-opt %s
module {
  func.func @sub_func () -> () {
    return
  }
  func.func @top() -> (){
    call @sub_func() : () -> ()
    %s = allo.create_op_handle "sub_func"
    allo.clone(@sub_func, %s)
    return
  }
}