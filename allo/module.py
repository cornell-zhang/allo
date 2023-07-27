# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import ctypes
import numpy as np
from hcl_mlir.ir import Module, UnitAttr, MemRefType, IntegerType, F32Type
from hcl_mlir.dialects import (
    hcl as hcl_d,
    func as func_d,
)
from hcl_mlir.passmanager import PassManager
from hcl_mlir.execution_engine import ExecutionEngine
from hcl_mlir.runtime import (
    get_ranked_memref_descriptor,
    make_nd_memref_descriptor,
    ranked_memref_to_numpy,
)
from .context import get_context, get_location
from .utils import np_type_to_str


class LLVMModule:
    def __init__(self, mod, top_func_name):
        # Copy the module to avoid modifying the original one
        with get_context() as ctx, get_location():
            self.module = Module.parse(str(mod), ctx)
            # find top func op
            func = None
            for op in self.module.body.operations:
                if isinstance(op, func_d.FuncOp) and op.name.value == top_func_name:
                    func = op
                    break
            if func is None:
                raise RuntimeError(
                    "No top-level function found in the built MLIR module"
                )
            func.attributes["llvm.emit_c_interface"] = UnitAttr.get()
            func.attributes["top"] = UnitAttr.get()
            self.top_func_type = func.type
            self.top_func_name = top_func_name
            # Remove .partition() annotation
            hcl_d.remove_stride_map(self.module)
            # Run through lowering passes
            pm = PassManager.parse(
                "func.func(convert-linalg-to-affine-loops),lower-affine,convert-scf-to-cf,convert-arith-to-llvm,convert-memref-to-llvm,convert-func-to-llvm,convert-cf-to-llvm,reconcile-unrealized-casts"
            )
            pm.run(self.module)
            # Add shared library
            if os.getenv("LLVM_BUILD_DIR") is not None:
                shared_libs = [
                    os.path.join(
                        os.getenv("LLVM_BUILD_DIR"), "lib", "libmlir_runner_utils.so"
                    ),
                    os.path.join(
                        os.getenv("LLVM_BUILD_DIR"), "lib", "libmlir_c_runner_utils.so"
                    ),
                ]
            else:
                shared_libs = None
            self.execution_engine = ExecutionEngine(
                self.module, opt_level=3, shared_libs=shared_libs
            )

    def __call__(self, *args):
        """
        Reference:
        * https://github.com/llvm/llvm-project/blob/llvmorg-15.0.0/mlir/test/python/execution_engine.py
        * https://github.com/llvm/llvm-project/blob/llvmorg-15.0.0/mlir/test/Integration/Dialect/SparseTensor/python/test_SpMM.py
        """
        input_types = self.top_func_type.inputs
        new_args = []
        for arg, in_type in zip(args, input_types):
            if not isinstance(arg, np.ndarray):
                if isinstance(arg, int):
                    if str(in_type) != "i32":
                        raise RuntimeError(
                            "Input type mismatch, expected i32, but got {}".format(
                                str(in_type)
                            )
                        )
                    c_int_p = ctypes.c_int * 1
                    new_args.append(c_int_p(arg))
                elif isinstance(arg, float):
                    if str(in_type) != "f32":
                        raise RuntimeError(
                            "Input type mismatch, expected f32, but got {}".format(
                                str(in_type)
                            )
                        )
                    c_float_p = ctypes.c_float * 1
                    new_args.append(c_float_p(arg))
                else:
                    raise RuntimeError("Unsupported input type")
            else:
                np_type = np_type_to_str(arg.dtype)
                target_type = str(MemRefType(in_type).element_type)
                if np_type != target_type:
                    raise RuntimeError(
                        "Input type mismatch: {} vs {}".format(np_type, target_type)
                    )
                new_args.append(
                    ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(arg)))
                )
        # TODO: only support one return value for now
        result_types = self.top_func_type.results
        if len(result_types) != 1:
            raise RuntimeError("Only support one return value for now")
        if MemRefType.isinstance(result_types[0]):
            result_type = MemRefType(result_types[0])
            shape = result_type.shape
            result_type = result_type.element_type
            if str(result_type) == "f32":
                dtype = ctypes.c_float
            elif str(result_type) == "f64":
                dtype = ctypes.c_double
            elif str(result_type) == "i32":
                dtype = ctypes.c_int32
            elif str(result_type) == "i64":
                dtype = ctypes.c_int64
            else:
                raise RuntimeError("Unsupported return type")
            return_desc = make_nd_memref_descriptor(len(shape), dtype)()
            return_tensor = ctypes.pointer(ctypes.pointer(return_desc))
        elif IntegerType.isinstance(result_types[0]):
            result_type = IntegerType(result_types[0])
            if str(result_type) == "i32":
                dtype = ctypes.c_int32
            elif str(result_type) == "i64":
                dtype = ctypes.c_int64
            else:
                raise RuntimeError("Unsupported return type")
            dtype_p = dtype * 1
            return_tensor = dtype_p(-1)
        elif F32Type.isinstance(result_types[0]):
            result_type = F32Type(result_types[0])
            dtype_p = ctypes.c_float * 1
            return_tensor = dtype_p(-1.0)
        else:
            raise RuntimeError("Unsupported return type")
        if MemRefType.isinstance(result_types[0]):
            self.execution_engine.invoke(self.top_func_name, return_tensor, *new_args)
            ret = ranked_memref_to_numpy(return_tensor[0])
        else:
            self.execution_engine.invoke(self.top_func_name, *new_args, return_tensor)
            ret = return_tensor[0]
        return ret
