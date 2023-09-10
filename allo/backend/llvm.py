# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=no-name-in-module, inconsistent-return-statements

import os
import ctypes
import numpy as np
from hcl_mlir.ir import (
    Context,
    Location,
    Module,
    UnitAttr,
)
from hcl_mlir.dialects import hcl as hcl_d
from hcl_mlir.passmanager import PassManager
from hcl_mlir.execution_engine import ExecutionEngine
from hcl_mlir.runtime import (
    get_ranked_memref_descriptor,
    make_nd_memref_descriptor,
    # ranked_memref_to_numpy,
    to_numpy,
)
from hcl_mlir.exceptions import DTypeWarning
from ..ir.transform import find_func_in_module
from ..passes import _mlir_lower_pipeline, decompose_library_function
from ..utils import (
    get_func_inputs_outputs,
    get_bitwidth_from_type,
    get_bitwidth_and_frac_from_fixed,
    get_clostest_pow2,
    ctype_map,
    np_supported_types,
    np_type_to_str,
    is_anywidth_int_type_and_not_np,
    handle_overflow,
    make_anywidth_numpy_array,
    struct_array_to_int_array,
    get_np_struct_type,
)


def ranked_memref_to_numpy(ranked_memref):
    """Converts ranked memrefs to numpy arrays."""
    # A temporary workaround for issue
    # https://discourse.llvm.org/t/setting-memref-elements-in-python-callback/72759
    contentPtr = ctypes.cast(
        ctypes.addressof(ranked_memref[0].aligned.contents)
        + ranked_memref[0].offset * ctypes.sizeof(ranked_memref[0].aligned.contents),
        type(ranked_memref[0].aligned),
    )
    np_arr = np.ctypeslib.as_array(contentPtr, shape=ranked_memref[0].shape)
    strided_arr = np.lib.stride_tricks.as_strided(
        np_arr,
        np.ctypeslib.as_array(ranked_memref[0].shape),
        np.ctypeslib.as_array(ranked_memref[0].strides) * np_arr.itemsize,
    )
    return to_numpy(strided_arr)


def invoke_mlir_parser(mod: str):
    with Context() as ctx, Location.unknown():
        hcl_d.register_dialect(ctx)
        module = Module.parse(str(mod), ctx)
    return module


class LLVMModule:
    def __init__(self, mod, top_func_name, ext_libs=None):
        # Copy the module to avoid modifying the original one
        with Context() as ctx:
            hcl_d.register_dialect(ctx)
            self.module = Module.parse(str(mod), ctx)
            self.top_func_name = top_func_name
            func = find_func_in_module(self.module, top_func_name)
            # Get input/output types
            self.in_types, self.out_types = get_func_inputs_outputs(func)
            self.module = decompose_library_function(self.module)
            # Start lowering
            _mlir_lower_pipeline(self.module, canonicalize=True, lower_linalg=True)
            # Remove .partition() annotation
            hcl_d.remove_stride_map(self.module)
            # Resolve FixedType
            hcl_d.lower_fixed_to_int(self.module)
            hcl_d.lower_bit_ops(self.module)
            # Run through lowering passes
            pm = PassManager.parse(
                "builtin.module("
                # used for lowering tensor.empty
                "empty-tensor-to-alloc-tensor,"
                # translate tensor dialect (virtual) to memref dialect (physical)
                "one-shot-bufferize{allow-return-allocs bufferize-function-boundaries},"
                # used for lowering memref.subview
                "expand-strided-metadata,"
                # common lowering passes
                "func.func(convert-linalg-to-affine-loops),lower-affine"
                ")"
            )
            pm.run(self.module.operation)
            # Attach necessary attributes
            func = find_func_in_module(self.module, top_func_name)
            if func is None:
                raise RuntimeError(
                    "No top-level function found in the built MLIR module"
                )
            func.attributes["llvm.emit_c_interface"] = UnitAttr.get()
            func.attributes["top"] = UnitAttr.get()
            # Final lowering
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
                shared_libs = []
            shared_libs += ext_libs if ext_libs is not None else []
            # opt_level should be set to 2 to avoid the following issue
            # https://github.com/cornell-zhang/allo/issues/72
            self.execution_engine = ExecutionEngine(
                self.module, opt_level=2, shared_libs=shared_libs
            )

    # pylint: disable=too-many-branches
    def __call__(self, *args):
        """
        Reference:
        * https://github.com/llvm/llvm-project/blob/llvmorg-15.0.0/mlir/test/python/execution_engine.py
        * https://github.com/llvm/llvm-project/blob/llvmorg-15.0.0/mlir/test/Integration/Dialect/SparseTensor/python/test_SpMM.py
        """
        input_types = self.in_types
        arg_ptrs = []
        new_args = []
        assert len(args) == len(
            input_types
        ), f"# of input arguments mismatch, got {len(args)} but expected {len(input_types)}"

        # 1. Construct argument pointers
        for arg, (target_in_type, shape) in zip(args, input_types):
            if len(shape) == 0:  # scalar
                if isinstance(arg, int):
                    if target_in_type != "i32":
                        DTypeWarning(
                            f"Input type mismatch: {target_in_type} vs i32. Please use NumPy array"
                            "to wrap the data to avoid possible result mismatch"
                        ).warn()
                    bitwidth = get_bitwidth_from_type(target_in_type)
                    pow2_width = max(get_clostest_pow2(bitwidth), 8)
                    signed = "i" if target_in_type.startswith("i") else "ui"
                    dtype = ctype_map[f"{signed}{pow2_width}"]
                    c_int_p = dtype * 1
                    arg_ptrs.append(c_int_p(arg))
                elif isinstance(arg, float):
                    if target_in_type != "f32":
                        DTypeWarning(
                            f"Input type mismatch: {target_in_type} vs f32. Please use NumPy array"
                            "to wrap the data to avoid possible result mismatch"
                        ).warn()
                    if target_in_type == "f32":
                        c_float_p = ctypes.c_float * 1
                    else:  # f64
                        c_float_p = ctypes.c_double * 1
                    arg_ptrs.append(c_float_p(arg))
                else:
                    raise RuntimeError(
                        "Unsupported input type. Please use NumPy array to wrap the data if other"
                        "data types are needed as inputs."
                    )
            else:
                np_type = np_type_to_str(arg.dtype)
                if np_type != target_in_type:
                    DTypeWarning(
                        f"Input type mismatch: {np_type} vs {target_in_type}"
                    ).warn()
                if is_anywidth_int_type_and_not_np(target_in_type):
                    bitwidth = get_bitwidth_from_type(target_in_type)
                    arg = handle_overflow(arg, bitwidth, target_in_type)
                    # This is to be compliant with MLIR's anywidth int type alignment
                    # e.g. i1-i8 -> int8
                    #      i9-i16 -> int16
                    #      i17-i32 -> int32
                    #      i33-i64 -> int64
                    #      i65-i128 -> int128
                    #      i129-i256 -> int256
                    # pylint: disable=redefined-variable-type
                    arg = make_anywidth_numpy_array(arg, bitwidth)
                elif target_in_type in np_supported_types:
                    target_np_type = np_supported_types[target_in_type]
                    if arg.dtype != target_np_type:
                        # avoid changing the address of the original array
                        arg = arg.astype(target_np_type)
                elif target_in_type.startswith("fixed") or target_in_type.startswith(
                    "ufixed"
                ):
                    arg = arg.astype(np.float64)
                    bitwidth, frac = get_bitwidth_and_frac_from_fixed(target_in_type)
                    arg = arg * (2**frac)
                    arg = handle_overflow(arg, bitwidth, target_in_type)
                    arg = make_anywidth_numpy_array(arg, bitwidth)
                else:
                    raise RuntimeError(
                        f"Unsupported input type: {target_in_type}, "
                        f"please use a supported type or wrap the scalar as an array"
                    )
                arg_ptrs.append(
                    ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(arg)))
                )
            new_args.append(arg)

        # 2. Construct return pointers
        # Need to verify the return variable is not the same as the input
        result_types = self.out_types
        if len(result_types) > 1:
            raise RuntimeError("Only support zero/one return value for now")
        # Returns as arguments
        if len(result_types) == 0:
            self.execution_engine.invoke(self.top_func_name, *arg_ptrs)
            for arg, new_arg, (target_in_type, shape) in zip(
                args, new_args, input_types
            ):
                if len(shape) > 0:
                    if is_anywidth_int_type_and_not_np(target_in_type):
                        bitwidth = get_bitwidth_from_type(target_in_type)
                        arg[:] = struct_array_to_int_array(
                            new_arg, bitwidth, target_in_type[0] == "i"
                        )
                    else:
                        arg[:] = new_arg
            return
        # Return inner variables
        result_type, shape = result_types[0]
        if len(shape) == 0:  # scalar
            if result_type in ctype_map:
                dtype = ctype_map[result_type]
            else:
                if result_type.startswith("fixed") or result_type.startswith("ufixed"):
                    raise RuntimeError("Not supported FixedType returns")
                DTypeWarning(
                    f"Return type {result_type} is not supported by native Python. "
                    "Please change another return type or use Numpy array to wrap the return value"
                ).warn()
                signed = "i" if result_type.startswith("i") else "ui"
                bitwidth = get_bitwidth_from_type(result_type)
                pow2_width = max(get_clostest_pow2(bitwidth), 8)
                dtype = ctype_map[f"{signed}{pow2_width}"]
            dtype_p = dtype * 1
            # -1/-1.0 is a placeholder
            return_ptr = dtype_p(-1 if not result_type in {"f32", "f64"} else 1.0)
        else:
            if result_type in ctype_map:
                dtype = ctype_map[result_type]
            elif result_type.startswith("i") or result_type.startswith("ui"):
                width = get_bitwidth_from_type(result_type)
                bitwidth = max(get_clostest_pow2(width), 8)
                dtype = np.ctypeslib.as_ctypes_type(get_np_struct_type(bitwidth))
            elif result_type.startswith("fixed") or result_type.startswith("ufixed"):
                bitwidth, _ = get_bitwidth_and_frac_from_fixed(result_type)
                bitwidth = max(get_clostest_pow2(bitwidth), 8)
                dtype = np.ctypeslib.as_ctypes_type(get_np_struct_type(bitwidth))
            else:
                raise RuntimeError("Unsupported return type")
            # Create an empty tensor
            return_desc = make_nd_memref_descriptor(len(shape), dtype)()
            return_ptr = ctypes.pointer(ctypes.pointer(return_desc))

        # 3. Invoke the function and return the result
        if len(shape) > 0:
            self.execution_engine.invoke(self.top_func_name, return_ptr, *arg_ptrs)
            ret = ranked_memref_to_numpy(return_ptr[0])
            if is_anywidth_int_type_and_not_np(result_type):
                bitwidth = get_bitwidth_from_type(result_type)
                ret = struct_array_to_int_array(ret, bitwidth, result_type[0] == "i")
            elif result_type.startswith("fixed") or result_type.startswith("ufixed"):
                bitwidth, frac = get_bitwidth_and_frac_from_fixed(result_type)
                ret = struct_array_to_int_array(
                    ret, bitwidth, result_type.startswith("fixed")
                )
                if result_type.startswith("fixed"):
                    ret = ret.astype(np.int64)
                else:
                    ret = ret.astype(np.uint64)
                ret = ret.astype(np.float64) / float(2**frac)
        else:
            self.execution_engine.invoke(self.top_func_name, *arg_ptrs, return_ptr)
            ret = return_ptr[0]
        return ret
