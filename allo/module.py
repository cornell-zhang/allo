# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=no-name-in-module, consider-using-with

import os
import io
import subprocess
import time
import ctypes
import numpy as np
from hcl_mlir.ir import (
    Context,
    Location,
    Module,
    UnitAttr,
    MemRefType,
    IntegerType,
    F32Type,
)
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
from .utils import np_type_to_str
from .report import parse_xml
from .runtime import run_process, copy_build_files


def invoke_mlir_parser(mod: str):
    with Context() as ctx, Location.unknown():
        hcl_d.register_dialect(ctx)
        module = Module.parse(str(mod), ctx)
    return module


class LLVMModule:
    def __init__(self, mod, top_func_name):
        # Copy the module to avoid modifying the original one
        with Context() as ctx:
            self.module = Module.parse(str(mod), ctx)
            # Remove .partition() annotation
            hcl_d.remove_stride_map(self.module)
            # Run through lowering passes
            pm = PassManager.parse(
                "builtin.module(one-shot-bufferize{allow-return-allocs bufferize-function-boundaries},func.func(convert-linalg-to-affine-loops),lower-affine)"
            )
            pm.run(self.module.operation)
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
            hcl_d.lower_hcl_to_llvm(self.module, ctx)
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
                            f"Input type mismatch, expected i32, but got {str(in_type)}"
                        )
                    c_int_p = ctypes.c_int * 1
                    new_args.append(c_int_p(arg))
                elif isinstance(arg, float):
                    if str(in_type) != "f32":
                        raise RuntimeError(
                            f"Input type mismatch, expected f32, but got {str(in_type)}"
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
                        f"Input type mismatch: {np_type} vs {target_type}"
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


class HLSModule:
    def __init__(self, mod, top_func_name, mode=None, project=None):
        self.module = mod
        self.top_func_name = top_func_name
        self.mode = mode
        self.project = project
        buf = io.StringIO()
        hcl_d.emit_vhls(self.module, buf)
        buf.seek(0)
        self.hls_code = buf.read()
        if project is not None:
            assert mode is not None, "mode must be specified when project is specified"
            copy_build_files(self.top_func_name, project, mode)
            with open(f"{project}/kernel.cpp", "w", encoding="utf-8") as outfile:
                outfile.write(self.hls_code)
            with open(f"{project}/host.cpp", "w", encoding="utf-8") as outfile:
                outfile.write("")

    def __repr__(self):
        if self.mode is None:
            return self.hls_code
        return f"HLSModule({self.top_func_name}, {self.mode}, {self.project})"

    def __call__(self, shell=True):
        platform = "vivado_hls"
        if platform in {"vivado_hls", "vitis_hls"}:
            assert (
                os.system(f"which {platform} >> /dev/null") == 0
            ), f"cannot find {platform} on system path"
            ver = run_process("g++ --version", r"\d\.\d\.\d")[0].split(".")
            assert (
                int(ver[0]) * 10 + int(ver[1]) >= 48
            ), f"g++ version too old {ver[0]}.{ver[1]}.{ver[2]}"

            cmd = f"cd {self.project}; make "
            if self.mode == "csim":
                cmd += "csim"
                out = run_process(cmd + " 2>&1")
                runtime = [k for k in out.split("\n") if "seconds" in k][0]
                print(
                    f"[{time.strftime('%H:%M:%S', time.gmtime())}] Simulation runtime {runtime}"
                )

            elif "csyn" in self.mode or self.mode == "custom" or self.mode == "debug":
                cmd += platform
                print(
                    f"[{time.strftime('%H:%M:%S', time.gmtime())}] Begin synthesizing project ..."
                )
                if shell:
                    subprocess.Popen(cmd, shell=True).wait()
                else:
                    subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).wait()
                if self.mode != "custom":
                    out = parse_xml(
                        self.project,
                        "Vivado HLS",
                        top=self.top_func_name,
                        print_flag=True,
                    )

            else:
                raise RuntimeError(f"{platform} does not support {self.mode} mode")
        else:
            raise RuntimeError("Not implemented")
