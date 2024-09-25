# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# RUN: %PYTHON %s

import io
from hcl_mlir.ir import Context, Module
from hcl_mlir.dialects import hcl as hcl_d


def test_codegen():
    mlir_code = """
    module {
        func.func @gemm(%A: memref<1024x512xf32>, %B: memref<512x1024xf32>, %C: memref<1024x1024xf32>)
        {
            affine.for %i = 0 to 1024 {
                affine.for %j = 0 to 1024 {
                    affine.for %k = 0 to 512 {
                        %a = affine.load %A[%i, %k] : memref<1024x512xf32>
                        %b = affine.load %B[%k, %j] : memref<512x1024xf32>
                        %c = affine.load %C[%i, %j] : memref<1024x1024xf32>
                        %prod = arith.mulf %a, %b : f32
                        %sum = arith.addf %prod, %c: f32
                        affine.store %sum, %C[%i, %j] : memref<1024x1024xf32>
                    } { loop_name = "k" }
                } { loop_name = "j" }
            } { loop_name = "i", op_name = "s" }
            return
        }
    }
    """
    ctx = Context()
    mod = Module.parse(mlir_code, ctx)
    buf = io.StringIO()
    res = hcl_d.emit_vhls(mod, buf)
    if res:
        buf.seek(0)
        hls_code = buf.read()
        print(hls_code)
        print("Done HLS code generation")
    else:
        raise RuntimeError("HLS codegen failed")


if __name__ == "__main__":
    test_codegen()
