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
from hcl_mlir.exceptions import DTypeWarning


np_supported_types = {
    "f32": np.float32,
    "f64": np.float64,
    "i8": np.int8,
    "i16": np.int16,
    "i32": np.int32,
    "i64": np.int64,
    "ui8": np.uint8,
    "ui16": np.uint16,
    "ui32": np.uint32,
    "ui64": np.uint64,
}


def np_type_to_str(dtype):
    return list(np_supported_types.keys())[
        list(np_supported_types.values()).index(dtype)
    ]


def get_clostest_pow2(n):
    # .bit_length() is a Python method
    return 1 << (n - 1).bit_length()


def get_np_pow2_type(bitwidth, signed=True):
    if bitwidth <= 8:
        return np.int8 if signed else np.uint8
    if bitwidth <= 16:
        return np.int16 if signed else np.uint16
    if bitwidth <= 32:
        return np.int32 if signed else np.uint32
    if bitwidth <= 64:
        return np.int64 if signed else np.uint64
    raise RuntimeError("Unsupported bitwidth")


def get_np_struct_type(bitwidth):
    n_bytes = int(np.ceil(bitwidth / 8))
    return np.dtype(
        {
            "names": [f"f{i}" for i in range(n_bytes)],
            # all set to unsigned byte
            "formats": ["u1"] * n_bytes,
            "offsets": list(range(n_bytes)),
            "itemsize": n_bytes,
        }
    )


def is_anywidth_int_type_and_not_np(dtype):
    return str(dtype) not in np_supported_types and (
        str(dtype).startswith("i") or str(dtype).startswith("u")
    )


def make_anywidth_numpy_array(array, bitwidth):
    """
    Converts a numpy array to any target bitwidth.
    ----------------
    Parameters:
    array: numpy.ndarray
        numpy array, can be any numpy native bitwidth, e.g. np.int64
    bitwidth: int
        target bitwidth e.g. 9, 31, 198
    ----------------
    Returns:
    numpy.ndarray
        numpy array with the target bitwidth
    """
    shape = array.shape
    sign_array = array >= 0
    avail_bytes = array.itemsize  # number of bytes of each element
    # The following code has several steps to convert the numpy array to have
    # the correct data type in order to create an MLIR constant tensor.
    # Since MLIR-NumPy Python interface only supports byte-addressable data types,
    # we need to change the data type of the array to have the minimum number of bytes
    # that can represent the target bitwidth.
    # e.g., hcl.const_tensor(arr, dtype=hcl.Int(20)) (6*6 array)
    #       which requires 20 bits (3 bytes) to represent each element
    # declaration: 6*6*i20
    # numpy input: 6*6*i64
    # 1. Decompose the original i32 or i64 array into a structured array of uint8
    #  -> decompose: 6*6*8*i8
    # pylint: disable=no-else-return
    if bitwidth == 1:
        return np.packbits(array, axis=None, bitorder="little")
    else:
        # Here we construct a customized NumPy dtype, "f0", "f1", "f2", etc.
        # are the field names, and the entire data type is `op.values.dtype`.
        # This can be viewed as a `union` type in C/C++.
        # Please refer to the documentation for more details:
        # https://numpy.org/doc/stable/reference/arrays.dtypes.html#specifying-and-constructing-data-types
        decomposed_np_dtype = np.dtype(
            (
                array.dtype,
                {f"f{i}": (np.uint8, i) for i in range(array.dtype.itemsize)},
            )
        )
        array = array.view(decomposed_np_dtype)
        # 2. Compose the uint8 array into a structured array of target bitwidth
        # This is done by taking the first several bytes of the uint8 array
        # "u1" means one unsigned byte, and "i1" means one signed byte
        # f0 is LSB, fn is MSB
        # [[(f0, f1, ..., f7), (f0, f1, ..., f7)], ...]
        n_bytes = int(np.ceil(bitwidth / 8))
        new_dtype = get_np_struct_type(bitwidth)
        # Take each byte as a separate array
        # [f0  [f1  [f2
        #  ..   ..   ..
        #  f0]  f1]  f2]
        _bytes = [array[f"f{i}"] for i in range(min(avail_bytes, n_bytes))]
        # sometimes the available bytes are not enough to represent the target bitwidth
        # so that we need to pad the array
        if avail_bytes < n_bytes:
            padding = np.where(sign_array, 0x00, 0xFF).astype(np.uint8)
            _bytes += [padding] * (n_bytes - avail_bytes)
        # Stack them together
        # -> compose: 6*6*3*i8
        array = np.stack(_bytes, axis=-1)
        # -> flatten: 108*i8
        array = array.flatten()
        # -> view: 36*i24
        array = array.view(np.dtype(new_dtype))
        # -> reshape: 6*6*i24
        array = array.reshape(shape)
        return array


def struct_array_to_int_array(array, dtype):
    """
    Converts a structured numpy array to back to an integer array.
    ----------------
    Parameters:
    array: numpy.ndarray
        A structured numpy array
    dtype: mlir.types
        The target MLIR element type
    ----------------
    Returns:
    numpy.ndarray
        numpy array in np.int64
    """
    # TODO: Sometimes we need to use Python native list to do the conversion
    #       since Numpy array cannot hold very long-bitwidth integer (>64)
    #       See: https://github.com/cornell-zhang/heterocl/pull/493
    # e.g., numpy input 6*6*i24
    # 1. Take each byte as a separate array
    # [f0  [f1  [f2
    #  ..   ..   ..
    #  f0]  f1]  f2]
    # -> unflatten: 6*6*3*i8
    bitwidth = dtype.width
    shape = array.shape
    signed = str(dtype).startswith("i")
    n_bytes = int(np.ceil(bitwidth / 8))
    if bitwidth > 64:
        raise RuntimeError("Cannot convert data with bitwidth > 64 to numpy array")
    target_bytes = max(get_clostest_pow2(bitwidth), 8) // 8
    if n_bytes == 1:
        _bytes = [array] if array.dtype == np.uint8 else [array["f0"]]
    else:
        _bytes = [array[f"f{i}"] for i in range(n_bytes)]
    # Take the negative sign part
    # Find the MSB
    bit_idx = (bitwidth - 1) % 8
    sign_array = (_bytes[-1] & (1 << bit_idx)) > 0
    # Need to also set _bytes[-1]
    if signed:
        _bytes[-1][sign_array] |= (0xFF << bit_idx) & 0xFF
    for _ in range(len(_bytes), target_bytes):
        if signed:
            sign = np.zeros_like(_bytes[0], dtype=np.uint8)
            sign[sign_array] = 0xFF
            _bytes.append(sign)
        else:
            _bytes.append(np.zeros_like(_bytes[0], dtype=np.uint8))
    # Stack them together
    # compose: 6*6*4*i8
    array = np.stack(_bytes, axis=-1)
    # -> flatten: 144*i8
    array = array.flatten()
    # -> view: 36*i32
    array = array.view(get_np_pow2_type(bitwidth))
    # -> reshape: 6*6*i32
    array = array.reshape(shape)
    return array


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
                "builtin.module(one-shot-bufferize{allow-return-allocs bufferize-function-boundaries},"
                "func.func(convert-linalg-to-affine-loops),lower-affine)"
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

    # pylint: disable=too-many-branches
    def __call__(self, *args):
        """
        Reference:
        * https://github.com/llvm/llvm-project/blob/llvmorg-15.0.0/mlir/test/python/execution_engine.py
        * https://github.com/llvm/llvm-project/blob/llvmorg-15.0.0/mlir/test/Integration/Dialect/SparseTensor/python/test_SpMM.py
        """
        input_types = self.top_func_type.inputs
        arg_ptrs = []
        new_args = []
        assert len(args) == len(
            input_types
        ), f"# of input arguments mismatch, got {len(args)} but expected {len(input_types)}"

        # 1. Construct argument pointers
        for arg, in_type in zip(args, input_types):
            if not isinstance(arg, np.ndarray):  # scalar
                if isinstance(arg, int):
                    assert (
                        str(in_type) == "i32"
                    ), f"Input type mismatch, expected i32, but got {str(in_type)}"
                    c_int_p = ctypes.c_int * 1
                    arg_ptrs.append(c_int_p(arg))
                elif isinstance(arg, float):
                    assert (
                        str(in_type) == "f32"
                    ), f"Input type mismatch, expected f32, but got {str(in_type)}"
                    c_float_p = ctypes.c_float * 1
                    arg_ptrs.append(c_float_p(arg))
                else:
                    raise RuntimeError("Unsupported input type")
            else:
                np_type = np_type_to_str(arg.dtype)
                target_type = MemRefType(in_type).element_type
                if np_type != str(target_type):
                    DTypeWarning(
                        f"Input type mismatch: {np_type} vs {str(target_type)}"
                    ).warn()
                # TODO: Handle overflow
                if is_anywidth_int_type_and_not_np(target_type):
                    # Int or UInt type
                    target_type = IntegerType(MemRefType(in_type).element_type)
                    # This is to be compliant with MLIR's anywidth int type alignment
                    # e.g. i1-i8 -> int8
                    #      i9-i16 -> int16
                    #      i17-i32 -> int32
                    #      i33-i64 -> int64
                    #      i65-i128 -> int128
                    #      i129-i256 -> int256
                    bitwidth = max(get_clostest_pow2(target_type.width), 8)
                    arg = make_anywidth_numpy_array(arg, bitwidth)
                elif str(target_type) in np_supported_types:
                    target_np_type = np_supported_types[str(target_type)]
                    if arg.dtype != target_np_type:
                        # avoid changing the address of the original array
                        arg = arg.astype(target_np_type)
                else:  # unsupported type (fixed type)
                    raise RuntimeError(
                        f"Unsupported input type: {str(target_type)}, "
                        f"please use a supported type or wrap the scalar as an array"
                    )
                arg_ptrs.append(
                    ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(arg)))
                )
            new_args.append(arg)

        # 2. Construct return pointers
        # Need to verify the return variable is not the same as the input
        result_types = self.top_func_type.results
        if len(result_types) > 1:
            raise RuntimeError("Only support zero/one return value for now")
        if len(result_types) == 0:
            self.execution_engine.invoke(self.top_func_name, *arg_ptrs)
            for i, (arg, new_arg) in enumerate(zip(args, new_args)):
                if isinstance(arg, np.ndarray):
                    target_type = MemRefType(input_types[i]).element_type
                    if is_anywidth_int_type_and_not_np(target_type):
                        arg[:] = struct_array_to_int_array(new_arg, target_type)
                    else:
                        arg[:] = new_arg
            return
        if MemRefType.isinstance(result_types[0]):
            result_type = MemRefType(result_types[0])
            shape = result_type.shape
            result_type = result_type.element_type
            if str(result_type) == "f32":
                dtype = ctypes.c_float
            elif str(result_type) == "f64":
                dtype = ctypes.c_double
            elif str(result_type) == "i8":
                dtype = ctypes.c_int8
            elif str(result_type) == "i16":
                dtype = ctypes.c_int16
            elif str(result_type) == "i32":
                dtype = ctypes.c_int32
            elif str(result_type) == "i64":
                dtype = ctypes.c_int64
            elif str(result_type).startswith("i"):
                bitwidth = max(get_clostest_pow2(result_type.width), 8)
                dtype = np.ctypeslib.as_ctypes_type(get_np_struct_type(bitwidth))
            else:
                raise RuntimeError("Unsupported return type")
            # Create an empty tensor
            return_desc = make_nd_memref_descriptor(len(shape), dtype)()
            return_ptr = ctypes.pointer(ctypes.pointer(return_desc))
        elif IntegerType.isinstance(result_types[0]):
            result_type = IntegerType(result_types[0])
            if str(result_type) == "i32":
                dtype = ctypes.c_int32
            elif str(result_type) == "i64":
                dtype = ctypes.c_int64
            else:
                raise RuntimeError(
                    "Unsupported return type, please wrap the scalar as an array"
                )
            dtype_p = dtype * 1
            # -1 is a placeholder
            return_ptr = dtype_p(-1)
        elif F32Type.isinstance(result_types[0]):
            result_type = F32Type(result_types[0])
            dtype_p = ctypes.c_float * 1
            # -1.0 is a placeholder
            return_ptr = dtype_p(-1.0)
        else:
            raise RuntimeError("Unsupported return type")

        # 3. Invoke the function and return the result
        if MemRefType.isinstance(result_types[0]):
            self.execution_engine.invoke(self.top_func_name, return_ptr, *arg_ptrs)
            ret = ranked_memref_to_numpy(return_ptr[0])
            result_type = MemRefType(result_types[0]).element_type
            if is_anywidth_int_type_and_not_np(result_type):
                ret = struct_array_to_int_array(ret, result_type)
        else:
            self.execution_engine.invoke(self.top_func_name, *arg_ptrs, return_ptr)
            ret = return_ptr[0]
        return ret
