# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# RUN: %PYTHON %s

import io

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

    res = allo_d.loop_transformation(mod)
    if res:
        print(str(mod))
        print("Done loop transformation!")
    else:
        raise RuntimeError("Loop transformation failed")

    buf = io.StringIO()
    res = allo_d.emit_vhls(mod, buf)
    if res:
        buf.seek(0)
        hls_code = buf.read()
        print(hls_code)
        print("Done HLS code generation")
    else:
        raise RuntimeError("HLS codegen failed")
