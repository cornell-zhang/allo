# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=no-name-in-module

from hcl_mlir.ir import (
    MemRefType,
    RankedTensorType,
    IntegerType,
    F32Type,
    F64Type,
)
from hcl_mlir.dialects import hcl as hcl_d


def get_bitwidth_from_type(dtype):
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
    if IntegerType.isinstance(dtype):
        return str(IntegerType(dtype)), tuple()
    if F32Type.isinstance(dtype):
        return str(F32Type(dtype)), tuple()
    if F64Type.isinstance(dtype):
        return str(F64Type(dtype)), tuple()
    if hcl_d.FixedType.isinstance(dtype):
        dtype = hcl_d.FixedType(dtype)
        width, frac = dtype.width, dtype.frac
        return f"fixed({width}, {frac})", tuple()
    if hcl_d.UFixedType.isinstance(dtype):
        dtype = hcl_d.UFixedType(dtype)
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
