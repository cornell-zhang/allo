# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=no-name-in-module

import ctypes
import numpy as np
from ._mlir.ir import (
    MemRefType,
    RankedTensorType,
    IntegerType,
    IndexType,
    F16Type,
    F32Type,
    F64Type,
)
from ._mlir.exceptions import DTypeWarning
from ._mlir.runtime import to_numpy
from ._mlir.dialects import allo as allo_d


np_supported_types = {
    "f16": np.float16,
    "f32": np.float32,
    "f64": np.float64,
    "i8": np.int8,
    "i16": np.int16,
    "i32": np.int32,
    "i64": np.int64,
    "ui1": np.bool_,
    "ui8": np.uint8,
    "ui16": np.uint16,
    "ui32": np.uint32,
    "ui64": np.uint64,
}


ctype_map = {
    # ctypes.c_float16 does not exist
    # similar implementation in _mlir/runtime/np_to_memref.py/F16
    "f16": ctypes.c_int16,
    "f32": ctypes.c_float,
    "f64": ctypes.c_double,
    "i8": ctypes.c_int8,
    "i16": ctypes.c_int16,
    "i32": ctypes.c_int32,
    "i64": ctypes.c_int64,
    "ui1": ctypes.c_bool,
    "ui8": ctypes.c_uint8,
    "ui16": ctypes.c_uint16,
    "ui32": ctypes.c_uint32,
    "ui64": ctypes.c_uint64,
}


def np_type_to_str(dtype):
    return list(np_supported_types.keys())[
        list(np_supported_types.values()).index(dtype)
    ]


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
        str(dtype).startswith("i") or str(dtype).startswith("ui")
    )


def get_clostest_pow2(n):
    # .bit_length() is a Python method
    return 1 << (n - 1).bit_length()


def get_bitwidth_from_type(dtype):
    if dtype == "index":
        return 64
    if dtype.startswith("i"):
        return int(dtype[1:])
    if dtype.startswith("ui"):
        return int(dtype[2:])
    if dtype.startswith("fixed") or dtype.startswith("ufixed"):
        return int(dtype.split(",")[0].split("(")[-1])
    if dtype.startswith("f"):
        return int(dtype[1:])
    raise RuntimeError("Unsupported type")


def get_bitwidth_and_frac_from_fixed(dtype):
    bitwidth, frac = dtype.split("(")[-1][:-1].split(",")
    return int(bitwidth), int(frac)


def get_signed_type_by_hint(dtype, hint):
    if hint == "u" and (dtype.startswith("i") or dtype.startswith("fixed")):
        return "u" + dtype
    return dtype


def get_mlir_dtype_from_str(dtype):
    if dtype.startswith("i"):
        bitwidth = get_bitwidth_from_type("i" + dtype[3:])
        return IntegerType.get_signless(bitwidth)
    if dtype.startswith("ui"):
        bitwidth = get_bitwidth_from_type("ui" + dtype[3:])
        return IntegerType.get_unsigned(bitwidth)
    if dtype.startswith("fixed"):
        bitwidth, frac = get_bitwidth_and_frac_from_fixed(dtype)
        return allo_d.FixedType.get(bitwidth, frac)
    if dtype.startswith("ufixed"):
        bitwidth, frac = get_bitwidth_and_frac_from_fixed(dtype)
        return allo_d.UFixedType.get(bitwidth, frac)
    if dtype.startswith("f"):
        bitwidth = get_bitwidth_from_type("f" + dtype[5:])
        if bitwidth == 32:
            return F32Type.get()
        if bitwidth == 64:
            return F64Type.get()
        raise RuntimeError("Unsupported type")
    raise RuntimeError("Unsupported type")


def get_dtype_and_shape_from_type(dtype):
    if MemRefType.isinstance(dtype):
        dtype = MemRefType(dtype)
        shape = dtype.shape
        ele_type, _ = get_dtype_and_shape_from_type(dtype.element_type)
        return ele_type, shape
    if RankedTensorType.isinstance(dtype):
        dtype = RankedTensorType(dtype)
        shape = dtype.shape
        ele_type, _ = get_dtype_and_shape_from_type(dtype.element_type)
        return ele_type, shape
    if IndexType.isinstance(dtype):
        return "index", tuple()
    if IntegerType.isinstance(dtype):
        return str(IntegerType(dtype)), tuple()
    if F16Type.isinstance(dtype):
        return str(F16Type(dtype)), tuple()
    if F32Type.isinstance(dtype):
        return str(F32Type(dtype)), tuple()
    if F64Type.isinstance(dtype):
        return str(F64Type(dtype)), tuple()
    if allo_d.FixedType.isinstance(dtype):
        dtype = allo_d.FixedType(dtype)
        width, frac = dtype.width, dtype.frac
        return f"fixed({width}, {frac})", tuple()
    if allo_d.UFixedType.isinstance(dtype):
        dtype = allo_d.UFixedType(dtype)
        width, frac = dtype.width, dtype.frac
        return f"ufixed({width}, {frac})", tuple()
    raise RuntimeError("Unsupported type")


def get_func_inputs_outputs(func):
    inputs = []
    in_hints = (
        func.attributes["itypes"].value
        if "itypes" in func.attributes
        else "_" * len(func.type.inputs)
    )
    for in_type, in_hint in zip(func.type.inputs, in_hints):
        dtype, shape = get_dtype_and_shape_from_type(in_type)
        in_type = get_signed_type_by_hint(dtype, in_hint)
        inputs.append((in_type, shape))
    outputs = []
    out_hints = (
        func.attributes["otypes"].value
        if "otypes" in func.attributes
        else "_" * len(func.type.results)
    )
    for out_type, out_hint in zip(func.type.results, out_hints):
        dtype, shape = get_dtype_and_shape_from_type(out_type)
        out_type = get_signed_type_by_hint(dtype, out_hint)
        outputs.append((out_type, shape))
    return inputs, outputs


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
    bitwidth = max(get_clostest_pow2(bitwidth), 8)
    shape = array.shape
    sign_array = array >= 0
    avail_bytes = array.itemsize  # number of bytes of each element
    # The following code has several steps to convert the numpy array to have
    # the correct data type in order to create an MLIR constant tensor.
    # Since MLIR-NumPy Python interface only supports byte-addressable data types,
    # we need to change the data type of the array to have the minimum number of bytes
    # that can represent the target bitwidth.
    # e.g., allo.const_tensor(arr, dtype=allo.Int(20)) (6*6 array)
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


def struct_array_to_int_array(array, bitwidth, signed=True):
    """
    Converts a structured numpy array to back to an integer array.
    ----------------
    Parameters:
    array: numpy.ndarray
        A structured numpy array
    bitwidth: int
        target bitwidth e.g. 9, 31, 198
    signed: bool
        whether the target type is signed or not
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
    shape = array.shape
    n_bytes = int(np.ceil(bitwidth / 8))
    if bitwidth > 64:
        DTypeWarning(
            "Coverting data with bitwidth > 64 to numpy array, "
            "which may lead to possible incorrect results."
        ).warn()
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
    if bitwidth <= 64:
        array = array.view(get_np_pow2_type(bitwidth))
    else:
        array = array.view(get_np_struct_type(bitwidth))
    # -> reshape: 6*6*i32
    array = array.reshape(shape)
    return array


def handle_overflow(np_array, bitwidth, dtype):
    if dtype.startswith("fixed") or dtype.startswith("ufixed"):
        # Round to nearest integer towards zero
        np_dtype = np.int64 if dtype.startswith("fixed") else np.uint64
        np_array = np.fix(np_array).astype(np_dtype)
    sb = 1 << bitwidth
    sb_limit = 1 << (bitwidth - 1)
    np_array = np_array % sb

    if dtype.startswith("fixed") or dtype.startswith("i"):

        def cast_func(x):
            return x if x < sb_limit else x - sb

        return np.vectorize(cast_func)(np_array)
    return np_array


def ranked_memref_to_numpy(ranked_memref):
    """Converts ranked memrefs to numpy arrays."""
    # A temporary workaround for issue
    # https://discourse.llvm.org/t/setting-memref-elements-in-python-callback/72759
    contentPtr = ctypes.cast(
        ctypes.addressof(ranked_memref.aligned.contents)
        + ranked_memref.offset * ctypes.sizeof(ranked_memref.aligned.contents),
        type(ranked_memref.aligned),
    )
    np_arr = np.ctypeslib.as_array(contentPtr, shape=ranked_memref.shape)
    strided_arr = np.lib.stride_tricks.as_strided(
        np_arr,
        np.ctypeslib.as_array(ranked_memref.shape),
        np.ctypeslib.as_array(ranked_memref.strides) * np_arr.itemsize,
    )
    return to_numpy(strided_arr)


def create_output_struct(memref_descriptors):
    fields = [
        (f"memref{i}", memref.__class__) for i, memref in enumerate(memref_descriptors)
    ]
    # Dynamically create and return the new class
    OutputStruct = type("OutputStruct", (ctypes.Structure,), {"_fields_": fields})
    out_struct = OutputStruct()
    for i, memref in enumerate(memref_descriptors):
        setattr(out_struct, f"memref{i}", memref)
    return out_struct


def extract_out_np_arrays_from_out_struct(out_struct_ptr_ptr, num_output):
    out_np_arrays = []
    for i in range(num_output):
        out_np_arrays.append(
            ranked_memref_to_numpy(getattr(out_struct_ptr_ptr[0][0], f"memref{i}"))
        )
    return out_np_arrays
