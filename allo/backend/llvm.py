# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=no-name-in-module, inconsistent-return-statements, too-many-function-args

import os
import ctypes
import numpy as np
from .._mlir.ir import (
    Context,
    Location,
    Module,
    UnitAttr,
)
from .._mlir.dialects import allo as allo_d
from .._mlir.passmanager import PassManager
from .._mlir.execution_engine import ExecutionEngine
from .._mlir.runtime import (
    get_ranked_memref_descriptor,
    make_nd_memref_descriptor,
)
from .._mlir.exceptions import DTypeWarning
from ..ir.transform import find_func_in_module
from ..passes import (
    _mlir_lower_pipeline,
    decompose_library_function,
    call_ext_libs_in_ptr,
)
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
    create_output_struct,
    extract_out_np_arrays_from_out_struct,
    ranked_memref_to_numpy,
)


def invoke_mlir_parser(mod: str):
    with Context() as ctx, Location.unknown():
        allo_d.register_dialect(ctx)
        module = Module.parse(str(mod), ctx)
    return module


class LLVMModule:
    def __init__(self, mod, top_func_name, ext_libs=None):
        # Copy the module to avoid modifying the original one
        with Context() as ctx:
            allo_d.register_dialect(ctx)
            self.module = Module.parse(str(mod), ctx)
            self.top_func_name = top_func_name
            func = find_func_in_module(self.module, top_func_name)
            ext_libs = [] if ext_libs is None else ext_libs
            # Get input/output types
            self.in_types, self.out_types = get_func_inputs_outputs(func)
            self.module = decompose_library_function(self.module)
            # Start lowering
            _mlir_lower_pipeline(self.module, canonicalize=True, lower_linalg=True)
            if len(ext_libs) > 0:
                call_ext_libs_in_ptr(self.module, ext_libs)
            # Remove .partition() annotation
            allo_d.remove_stride_map(self.module)
            # Lower composite (struct) types
            allo_d.lower_composite_type(self.module)
            # Resolve FixedType
            allo_d.lower_fixed_to_int(self.module)
            allo_d.lower_bit_ops(self.module)
            # Run through lowering passes
            pm = PassManager.parse(
                "builtin.module("
                # used for lowering tensor.empty
                "empty-tensor-to-alloc-tensor,"
                # translate tensor dialect (virtual) to memref dialect (physical)
                "one-shot-bufferize{bufferize-function-boundaries},"
                # used for lowering memref.subview
                "expand-strided-metadata,"
                # common lowering passes
                "func.func(convert-linalg-to-affine-loops),lower-affine"
                ")"
            )
            pm.run(self.module.operation)
            self.intermediate_module = self.module.operation.clone()
            # Attach necessary attributes
            func = find_func_in_module(self.module, top_func_name)
            if func is None:
                raise RuntimeError(
                    "No top-level function found in the built MLIR module"
                )
            func.attributes["llvm.emit_c_interface"] = UnitAttr.get()
            func.attributes["top"] = UnitAttr.get()
            # Final lowering
            allo_d.lower_allo_to_llvm(self.module, ctx)
            pm = PassManager.parse("builtin.module(reconcile-unrealized-casts)")
            pm.run(self.module.operation)
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
            shared_libs += [lib.compile_shared_lib() for lib in ext_libs]
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
                            " to wrap the data to avoid possible result mismatch"
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
                            " to wrap the data to avoid possible result mismatch"
                        ).warn()
                    if target_in_type == "f16":
                        c_float_p = ctypes.c_int16 * 1
                        arg = np.float16(arg).view(np.int16)
                    elif target_in_type == "f32":
                        c_float_p = ctypes.c_float * 1
                    else:  # f64
                        c_float_p = ctypes.c_double * 1
                    arg_ptrs.append(c_float_p(arg))
                else:
                    raise RuntimeError(
                        "Unsupported input type. Please use NumPy array to wrap the data if other"
                        " data types are needed as inputs."
                    )
            else:  # memref
                if not arg.flags["C_CONTIGUOUS"]:
                    raise RuntimeError(
                        "The input data is not contiguous. Please use np.ascontiguousarray to change the layout first."
                    )
                if not isinstance(arg.dtype, np.dtypes.VoidDType):
                    np_type = np_type_to_str(arg.dtype)
                    if np_type != target_in_type:
                        DTypeWarning(
                            f"Input type mismatch: {np_type} vs {target_in_type}"
                        ).warn()
                if is_anywidth_int_type_and_not_np(target_in_type):
                    bitwidth = get_bitwidth_from_type(target_in_type)
                    # This is to be compliant with MLIR's anywidth int type alignment
                    # e.g. i1-i8 -> int8
                    #      i9-i16 -> int16
                    #      i17-i32 -> int32
                    #      i33-i64 -> int64
                    #      i65-i128 -> int128
                    #      i129-i256 -> int256
                    # pylint: disable=redefined-variable-type
                    if bitwidth <= 64:
                        arg = handle_overflow(arg, bitwidth, target_in_type)
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
        # Returns as arguments: no return value from the top function
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
        # Return inner variables: return one or more values allocated inside kernel
        # For two or more return values, llvm.emit_c_interface will return a struct
        # Therefore, for functions that return values, we need to separate two cases:
        # 1. return one value: no need to create a struct
        # 2. return two or more values: need to create a struct
        # In any case, we prepare a pointer of pointer to the return object
        # which is ready to be passed to the invoke function.
        if len(result_types) == 1:  # exactly one return value
            result_type, shape = result_types[0]
            if len(shape) == 0:  # scalar
                if result_type in ctype_map:
                    dtype = ctype_map[result_type]
                else:
                    if result_type.startswith("fixed") or result_type.startswith(
                        "ufixed"
                    ):
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
            else:  # memref
                if result_type in ctype_map:
                    dtype = ctype_map[result_type]
                elif result_type.startswith("i") or result_type.startswith("ui"):
                    width = get_bitwidth_from_type(result_type)
                    bitwidth = max(get_clostest_pow2(width), 8)
                    dtype = np.ctypeslib.as_ctypes_type(get_np_struct_type(bitwidth))
                elif result_type.startswith("fixed") or result_type.startswith(
                    "ufixed"
                ):
                    bitwidth, _ = get_bitwidth_and_frac_from_fixed(result_type)
                    bitwidth = max(get_clostest_pow2(bitwidth), 8)
                    dtype = np.ctypeslib.as_ctypes_type(get_np_struct_type(bitwidth))
                else:
                    raise RuntimeError("Unsupported return type")
                # Create an empty tensor
                return_desc = make_nd_memref_descriptor(len(shape), dtype)()
                return_ptr = ctypes.pointer(ctypes.pointer(return_desc))
        else:  # multiple return values
            # we assume all return values are memrefs
            out_memref_descs = []
            for elt_res_type, elt_shape in result_types:
                if len(elt_shape) == 0:
                    raise RuntimeError(
                        "When returning multiple values, we only support all tensors."
                    )
                if elt_res_type in ctype_map:
                    dtype = ctype_map[elt_res_type]
                elif elt_res_type.startswith("i") or elt_res_type.startswith("ui"):
                    width = get_bitwidth_from_type(elt_res_type)
                    bitwidth = max(get_clostest_pow2(width), 8)
                    dtype = np.ctypeslib.as_ctypes_type(get_np_struct_type(bitwidth))
                elif elt_res_type.startswith("fixed") or elt_res_type.startswith(
                    "ufixed"
                ):
                    bitwidth, _ = get_bitwidth_and_frac_from_fixed(elt_res_type)
                    bitwidth = max(get_clostest_pow2(bitwidth), 8)
                    dtype = np.ctypeslib.as_ctypes_type(get_np_struct_type(bitwidth))
                else:
                    raise RuntimeError("Unsupported return type")
                # Create an empty tensor
                return_desc = make_nd_memref_descriptor(len(elt_shape), dtype)()
                out_memref_descs.append(return_desc)
            # Create a struct
            out_struct = create_output_struct(out_memref_descs)
            return_ptr = ctypes.pointer(ctypes.pointer(out_struct))

        # 3. Invoke the function and return the result
        if len(result_types) == 1:
            result_type, shape = result_types[0]
            if len(shape) > 0:  # single return, memref
                # INVOKE
                self.execution_engine.invoke(self.top_func_name, return_ptr, *arg_ptrs)
                ret = ranked_memref_to_numpy(return_ptr[0][0])
                if is_anywidth_int_type_and_not_np(result_type):
                    bitwidth = get_bitwidth_from_type(result_type)
                    ret = struct_array_to_int_array(
                        ret, bitwidth, result_type[0] == "i"
                    )
                elif result_type == "f16":
                    ret = np.array(ret, dtype=np.int16).view(np.float16)
                elif result_type.startswith("fixed") or result_type.startswith(
                    "ufixed"
                ):
                    bitwidth, frac = get_bitwidth_and_frac_from_fixed(result_type)
                    ret = struct_array_to_int_array(
                        ret, bitwidth, result_type.startswith("fixed")
                    )
                    if result_type.startswith("fixed"):
                        ret = ret.astype(np.int64)
                    else:
                        ret = ret.astype(np.uint64)
                    ret = ret.astype(np.float64) / float(2**frac)
            else:  # single return, scalar
                # INVOKE
                self.execution_engine.invoke(self.top_func_name, *arg_ptrs, return_ptr)
                ret = return_ptr[0]
                if result_type == "f16":
                    ret = np.int16(ret).view(np.float16)
        else:  # multiple returns, assume all memref
            # INVOKE
            self.execution_engine.invoke(self.top_func_name, return_ptr, *arg_ptrs)
            ret_raw_np = extract_out_np_arrays_from_out_struct(
                return_ptr, len(result_types)
            )
            # pylint: disable=redefined-variable-type
            ret = []
            for np_arr, res_type in zip(
                ret_raw_np, [res_type for res_type, _ in result_types]
            ):
                if is_anywidth_int_type_and_not_np(res_type):
                    bitwidth = get_bitwidth_from_type(res_type)
                    ret_i = struct_array_to_int_array(
                        np_arr, bitwidth, res_type[0] == "i"
                    )
                elif res_type == "f16":
                    ret_i = np.array(np_arr, dtype=np.int16).view(np.float16)
                elif res_type.startswith("fixed") or res_type.startswith("ufixed"):
                    bitwidth, frac = get_bitwidth_and_frac_from_fixed(res_type)
                    ret_i = struct_array_to_int_array(
                        np_arr, bitwidth, res_type.startswith("fixed")
                    )
                    if res_type.startswith("fixed"):
                        ret_i = ret.astype(np.int64)
                    else:
                        ret_i = ret.astype(np.uint64)
                    ret_i = ret_i.astype(np.float64) / float(2**frac)
                else:
                    ret_i = np_arr
                ret.append(ret_i)
        return ret
