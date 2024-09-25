# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# RUN: %PYTHON %s

from hcl_mlir.ir import *
from hcl_mlir.dialects import hcl as hcl_d

with Context() as ctx:
    hcl_d.register_dialect()
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

    res = hcl_d.lower_hcl_to_llvm(mod, ctx)
    if res:
        print(str(mod))
        print("Done LLVM conversion")
    else:
        raise RuntimeError("LLVM pass error")
