# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# RUN: %PYTHON %s

from allo_mlir.ir import *
from allo_mlir.dialects import allo as allo_d

with Context() as ctx:
    allo_d.register_dialect()
    print("Registration done!")

    mod = Module.parse(
        """
        func.func @top () -> () {
            %0 = arith.constant 2 : i32
            %1 = arith.addi %0, %0 : i32
            return
        }
        """
    )
    print(str(mod))
    print("Done module parsing!")

    res = allo_d.lower_allo_to_llvm(mod, ctx)
    if res:
        print(str(mod))
        print("Done LLVM conversion")
    else:
        raise RuntimeError("LLVM pass error")
