# pylint: disable=import-error, no-name-in-module, c-extension-no-member, too-many-nested-blocks, consider-using-enumerate, consider-using-namedtuple-or-dataclass
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import re
import os
from dataclasses import dataclass
import numpy as np

import aie.ir as aie_ir
import allo._mlir._mlir_libs._mlir as allo_ir
from ..utils import format_str, format_code
from ...memory import DTensor
from .external_kernel import ExternalModule, ExternalModuleBase
from ..._mlir.dialects import (
    allo as allo_d,
    arith as allo_arith_d,
    func as allo_func_d,
)

from ..._mlir.ir import (
    MemRefType,
    InsertionPoint,
    FlatSymbolRefAttr,
    StringAttr,
    Type,
    Context,
    IntegerType,
    F16Type,
    F32Type,
    BF16Type,
)


# ############################################################
# Configurations
# ############################################################
@dataclass(frozen=True)
class Config:
    # https://riallto.ai/notebooks/3_2_Ryzenai_capabilities.html#interface-tile-properties
    COMPUTE_MAX_SEND = 2
    COMPUTE_MAX_RECV = 2
    MEM_MAX_SEND = 6
    MEM_MAX_RECV = 6
    SHIM_MAX_SEND = 2
    SHIM_MAX_RECV = 2
    # https://github.com/Xilinx/mlir-aie/blob/46bb8c25967f173eebe56056661be226b3933a14/programming_guide/section-2/section-2d/DMATasks.md#best-practices-for-data-movement-and-synchronization-with-npu_dma_memcpy_nd
    DMA_MAX_BDS = 16

    # https://github.com/Xilinx/mlir-aie/blob/v1.0/lib/Dialect/AIEX/IR/AIEXDialect.cpp#L233
    SHIM_DMA_HARDWARE_MAX_SIZES = [64, -1, 1024, 1024]
    # TODO: other dma size/stride constrain

    # fixme: some hyper-parameters, can be optimized
    IO_TILE_LOSE_FACTOR = 4
    COMPUTE_TILE_WITH_SHARED_MEMORY = 2
    LOCAL_CODE_OFFSET = 100
    GLOBAL_CODE_OFFSET = 10000


# reference: https://github.com/Xilinx/mlir-aie/blob/v1.0/docs/Devices.md
device_config_map = {
    "npu1": {"mesh": (4, 5), "mem_tile_num": 5, "shim_tile_num": 4},
    "npu1_4col": {"mesh": (4, 4), "mem_tile_num": 4, "shim_tile_num": 4},
    "npu1_3col": {"mesh": (4, 3), "mem_tile_num": 3, "shim_tile_num": 3},
    "npu1_2col": {"mesh": (4, 2), "mem_tile_num": 2, "shim_tile_num": 2},
    "npu1_1col": {"mesh": (4, 1), "mem_tile_num": 1, "shim_tile_num": 1},
    "npu2": {"mesh": (4, 8), "mem_tile_num": 8, "shim_tile_num": 8},
    "npu2_7col": {"mesh": (4, 7), "mem_tile_num": 7, "shim_tile_num": 7},
    "npu2_6col": {"mesh": (4, 6), "mem_tile_num": 6, "shim_tile_num": 6},
    "npu2_5col": {"mesh": (4, 5), "mem_tile_num": 5, "shim_tile_num": 5},
    "npu2_4col": {"mesh": (4, 4), "mem_tile_num": 4, "shim_tile_num": 4},
    "npu2_3col": {"mesh": (4, 3), "mem_tile_num": 3, "shim_tile_num": 3},
    "npu2_2col": {"mesh": (4, 2), "mem_tile_num": 2, "shim_tile_num": 2},
    "npu2_1col": {"mesh": (4, 1), "mem_tile_num": 1, "shim_tile_num": 1},
}


# ############################################################
# MLIR Code Generation
# ############################################################


@dataclass(frozen=True)
class StreamType:
    depth: int
    shape: list[int]
    dtype: str

    def __str__(self):
        return f"({self.dtype} {self.shape}, depth={self.depth})"

    def __repr__(self):
        return self.__str__()


class Stream:
    """
    Allo Stream class
    """

    def __init__(self, name: str):
        self.name = name
        self.type_str = None
        self.type: StreamType = None
        self.allo_element_type: Type = None  # element type in allo context
        self.is_tensor = False  # whether the stream carries tensor data

        self.src: str = None  # source tile of the stream
        self.dst: str = None  # destination tile of the stream

    def set_element_type(self, type_str: str, context: Context):
        """
        Set the element type of the stream from a type string.
        This function parses the type string and extracts the data shape and dtype.

        Args:
            - type_str (str): The IR type string
            - context (Context): The current allo MLIR context used for constructing types
        """
        if self.type is not None:
            assert type_str == self.type_str
            return
        self.type_str = type_str
        match = re.match(r"!allo\.stream<([^,]+),\s*(\d+)>", type_str)
        shape: list[int] = None
        dtype: str = None
        if match:
            with context, allo_ir.ir.Location.unknown():
                element_type_str = match.group(1)
                depth = int(match.group(2))
                memref_match = re.match(
                    r"memref<([0-9x\?]*)x?([a-z0-9]+)>", element_type_str
                )
                if memref_match:
                    shape_part = memref_match.group(1)
                    dtype = memref_match.group(2)
                    if shape_part == "":
                        shape = []
                    else:
                        shape = [
                            -1 if dim == "?" else int(dim)
                            for dim in shape_part.split("x")
                            if dim
                        ]
                else:
                    type_match = re.match(r"([a-z]+[0-9]*)", element_type_str)
                    if type_match:
                        shape, dtype = [], element_type_str
                    else:
                        raise ValueError(f"Invalid stream type {type_str}.")

                def get_element_allo_type(dtype_str: str) -> Type:
                    if dtype_str == "i32":
                        return IntegerType.get_signless(32)
                    if dtype_str == "i16":
                        return IntegerType.get_signless(16)
                    if dtype_str == "i8":
                        return IntegerType.get_signless(8)
                    if dtype_str == "f32":
                        return F32Type.get()
                    if dtype_str == "f16":
                        return F16Type.get()
                    if dtype_str == "bf16":
                        return BF16Type.get()
                    raise ValueError(f"Unsupported dtype: {dtype_str}")

                self.allo_element_type = MemRefType.get(
                    shape,
                    get_element_allo_type(dtype),
                )
                self.type = StreamType(depth, shape, dtype)
                self.is_tensor = len(shape) > 0
        else:
            raise ValueError(f"Invalid stream type {type_str}.")

    def __str__(self):
        return f"Stream (name={self.name}, dtype={self.allo_element_type}, is_tensor={self.is_tensor}, src={self.src}, dst={self.dst})"


@dataclass
class Argument:
    """
    Represents an argument to a function, either a DTensor or a Stream.
    """

    dtensor: DTensor
    stream: Stream


aie_ctype_map = {
    "bf16": "std::bfloat16_t",
    "f32": "float",
    "f64": "double",
    "i8": "int8_t",
    "i16": "short",
    "i32": "int",
    "i64": "long",
    "i128": "__int128_t",  # unverified
    "ui1": "bool",
    "ui8": "uint8_t",
    "ui16": "unsigned short",
    "ui32": "unsigned int",
    "ui64": "unsigned long",
}

aie_external_kernel_ctype_map = {
    "bf16": "bfloat16",
    "f32": "float",
    "f64": "double",
    "i8": "int8_t",
    "i16": "short",
    "i32": "int",
    "i64": "long",
    "i128": "__int128_t",  # unverified
    "ui1": "bool",
    "ui8": "uint8_t",
    "ui16": "unsigned short",
    "ui32": "unsigned int",
    "ui64": "unsigned long",
}


# aie::mmul size for different data type and different architectures used in library MatMul kernels
#   - aie2 kernel: https://github.com/Xilinx/mlir-aie/blob/v1.0/aie_kernels/aie2/mm.cc
#   - aie2p kernel: https://github.com/Xilinx/mlir-aie/blob/v1.0/aie_kernels/aie2p/mm.cc
matmul_external_kernel_config_map = {
    ("i8", "i8"): {"aie2": (4, 8, 8), "aie2p": (8, 8, 8)},
    ("i8", "i16"): {"aie2": (4, 8, 8), "aie2p": (8, 8, 8)},
    ("i8", "i32"): {"aie2": (4, 8, 8), "aie2p": (8, 8, 8)},
    ("i16", "i16"): {"aie2": (4, 4, 4), "aie2p": (4, 4, 8)},
    ("i16", "i32"): {"aie2": (4, 4, 4), "aie2p": (4, 4, 8)},
    ("bf16", "bf16"): {"aie2": (4, 8, 4), "aie2p": (8, 8, 8)},
    ("bf16", "f32"): {"aie2": (4, 8, 4), "aie2p": (8, 8, 8)},
}


def parse_kernel_name(name: str):
    match = re.match(r"(.+?)(_\d+(?:_\d+)*)$", name)
    if not match:
        raise ValueError(f"Invalid format: {name}")

    prefix = match.group(1).rstrip("_")
    indexs = tuple(int(n) for n in match.group(2).split("_") if n != "")
    return prefix, indexs


def collect_op_by_name(root, target: str) -> list:
    collected_op = []

    def collect(op):
        if op.name == target:
            collected_op.append(op.operation)
            return
        for region in op.regions:
            for block in region.blocks:
                for inner_op in block.operations:
                    collect(inner_op)

    collect(root)
    return collected_op


def inject_external_kernels(
    module: allo_ir.ir.Module,
    top_function_name,
    external_kernel_lib: dict[str, ExternalModule],
    lib_dir: str = "aie2",
) -> tuple[dict[str, bool], dict[str, ExternalModuleBase], set[str]]:
    """
    Inject external kernels for compute cores.
    TODO: is it possible to use cpp pass to inject?
            Ideally, we may want to pass the rewrite rule of each external kernel
            and use rewriter to perform pattern matching and replacement.

    For each top-level (non-private, non-top) function in the module, the function scans
    its operations. When it detects vector operations (`linalg.add` or `linalg.mul`) or
    matrix multiplications (`linalg.matmul`), it replaces them with external kernel calls
    and generates corresponding C++ kernel code snippets.

    Returns:
        - use_external_kernels: A mapping from function names to a boolean flag indicating
                                whether an external kernel was injected in that function.
        - injected_external_kernels: A dictionary mapping kernel names to tuples of external code
                            strings (C++ code and preprocessor defines).
        - include_src: A set of C++ include directives needed for the external kernels.
    """
    use_external_kernels: dict[str, bool] = {}
    injected_external_kernels: dict[str, ExternalModuleBase] = {}
    include_src: set[str] = set()

    def inject_external_kernels_recursive(operations, df_function_name: str):
        for op in operations:
            # 1. customized external kernel
            if isinstance(op, allo_func_d.CallOp):
                use_external_kernels[df_function_name] = True
                callee_name = op.callee.value
                # register external kernel
                if callee_name in injected_external_kernels:
                    continue
                external_module = external_kernel_lib[callee_name]
                assert external_module is not None, "external module not found"
                include_src.add(f'#include "{external_module.filename}"\n')
                injected_external_kernels[callee_name] = external_module
                continue
            # 2. builtin external kernel
            input_idx, output_idx = [], []
            kernel_code, kernel_header = "", ""
            call_builtin = False
            # fill with zero
            if op.operation.name == "linalg.fill" and isinstance(
                op.inputs[0].owner.opview, allo_arith_d.ConstantOp
            ):
                if (
                    op.inputs[0].owner.opview.literal_value == 0
                    and len(op.outputs[0].type.shape) <= 2
                ):
                    M = (
                        1
                        if len(op.outputs[0].type.shape) < 2
                        else op.outputs[0].type.shape[0]
                    )
                    N = op.outputs[0].type.shape[-1]
                    dtype = str(op.outputs[0].type.element_type)
                    ctype = aie_external_kernel_ctype_map[dtype]
                    include_src.add(f'#include "{lib_dir}/zero.cc"\n')
                    use_external_kernels[df_function_name] = True
                    kernel_name = f"fill_zeros_{dtype}_{M}_{N}_vector"
                    kernel_code += f"void {kernel_name}({ctype} *A)"
                    kernel_code += " {\n"
                    kernel_code += f"  zero_vectorized<{ctype}, {M}, {N}>(A);\n"
                    kernel_code += "}\n\n"
                    call_builtin = True
                    output_idx.append(0)
                    operands = [op.outputs[0]]
            # vec add/mul
            elif op.operation.name in {"linalg.add", "linalg.mul"}:
                op_name = op.operation.name.split(".")[1]
                include_src.add(f'#include "aie2/{op_name}.cc"\n')
                dtype = str(op.inputs[0].type.element_type)
                ctype = aie_external_kernel_ctype_map[dtype]
                kernel_name = f"{op_name}_{dtype}_vector"
                use_external_kernels[df_function_name] = True
                kernel_code += (
                    f"void {kernel_name}({ctype} *A_in, {ctype} *B_in, {ctype} *C_out)"
                )
                kernel_code += " {\n"
                kernel_code += f"  eltwise_v{op_name}<{ctype}, {ctype}, {np.prod(op.inputs[0].type.shape)}>(A_in, B_in, C_out);\n"
                kernel_code += "}\n\n"
                input_idx.extend([0, 1])
                output_idx.append(2)
                call_builtin = True
                operands = [
                    op.inputs[0],
                    op.inputs[1],
                    op.outputs[0],
                ]
            # matmul
            elif op.operation.name == "linalg.matmul":
                M, K = MemRefType(op.inputs[0].type).shape
                _, N = MemRefType(op.inputs[1].type).shape
                dtype = str(op.inputs[0].type.element_type)
                out_dtype = str(op.outputs[0].type.element_type)
                if (dtype, out_dtype) in matmul_external_kernel_config_map:
                    include_src.add('#include "mm.cc"\n')
                    use_external_kernels[df_function_name] = True
                    kernel_header += f"#define DIM_M {M}\n"
                    kernel_header += f"#define DIM_N {N}\n"
                    kernel_header += f"#define DIM_K {K}\n"
                    kernel_header += f"#define {dtype}_{out_dtype}_ONLY\n"
                    input_idx.extend([0, 1])
                    output_idx.append(2)
                    call_builtin = True
                    kernel_name = f"matmul_scalar_{dtype}_{out_dtype}"
                    operands = [
                        op.inputs[0],
                        op.inputs[1],
                        op.outputs[0],
                    ]
            if call_builtin:
                # replace operation
                call_op = allo_func_d.CallOp(
                    [],
                    FlatSymbolRefAttr.get(kernel_name),
                    operands,
                    ip=InsertionPoint(op),
                )
                call_op.attributes["lib"] = StringAttr.get(kernel_name)
                op.erase()
                # register external kernel
                if kernel_name in injected_external_kernels:
                    continue
                injected_external_kernels[kernel_name] = ExternalModuleBase(
                    kernel_name,
                    input_idx,
                    output_idx,
                    kernel_code,
                    kernel_header,
                )
                operand_types = [x.type for x in operands]
                func_type = allo_func_d.FunctionType.get(
                    operand_types,
                    [],
                )
                kernel = allo_func_d.FuncOp(
                    kernel_name,
                    func_type,
                    ip=InsertionPoint(func),
                )
                kernel.attributes["sym_visibility"] = StringAttr.get("private")
            else:
                for region in op.regions:
                    for block in region.blocks:
                        inject_external_kernels_recursive(
                            block.operations, df_function_name
                        )

    with module.context, allo_ir.ir.Location.unknown():
        for func in module.body.operations:
            if isinstance(func, allo_func_d.FuncOp) and (
                "sym_visibility" not in func.attributes
                or func.attributes["sym_visibility"].value != "private"
            ):
                if func.attributes["sym_name"].value != top_function_name:
                    func_name: str = func.attributes["sym_name"].value
                    use_external_kernels[func_name] = False
                    for block in func.regions[0].blocks:
                        inject_external_kernels_recursive(block.operations, func_name)
    return (
        use_external_kernels,
        injected_external_kernels,
        include_src,
    )


def get_df_kernels(module: allo_ir.ir.Module) -> list[allo_func_d.FuncOp]:
    df_kernels = []
    for func in module.body.operations:
        if isinstance(func, allo_func_d.FuncOp) and "df.kernel" in func.attributes:
            df_kernels.append(func)
    return df_kernels


def classify_aie_functions_experimental(
    module: allo_ir.ir.Module, top_function_name: str
) -> tuple[allo_func_d.FuncOp, list[allo_func_d.FuncOp], list[allo_func_d.FuncOp]]:
    """
    Classify the functions in allo module as
        - top
        - compute core functions
        - external kernel functions
    """
    top_func: allo_func_d.FuncOp = None
    core_funcs: list[allo_func_d.FuncOp] = []
    external_funcs: list[allo_func_d.FuncOp] = []
    with module.context, allo_ir.ir.Location.unknown():
        for func in module.body.operations:
            if isinstance(func, allo_func_d.FuncOp):
                if (
                    "sym_visibility" in func.attributes
                    and func.attributes["sym_visibility"].value == "private"
                ):
                    external_funcs.append(func)
                elif func.attributes["sym_name"].value == top_function_name:
                    top_func = func
                elif "df.kernel" in func.attributes:
                    core_funcs.append(func)
                else:
                    raise ValueError(
                        f"Unknown function type: {func.attributes['sym_name'].value}"
                    )
    return top_func, core_funcs, external_funcs


def classify_aie_functions(
    module: allo_ir.ir.Module, top_function_name: str
) -> tuple[
    allo_func_d.FuncOp, dict[str, list[allo_func_d.FuncOp]], list[allo_func_d.FuncOp]
]:
    """
    Classify the functions in allo module as
        - top
        - compute core functions
        - external kernel functions
    """
    # top function
    top_func: allo_func_d.FuncOp = None
    # core functions
    core_func_groups: dict[str, list[allo_func_d.FuncOp]] = {}
    # external functions
    external_funcs: list[allo_func_d.FuncOp] = []
    with module.context, allo_ir.ir.Location.unknown():
        for func in module.body.operations:
            if isinstance(func, allo_func_d.FuncOp):
                if (
                    "sym_visibility" in func.attributes
                    and func.attributes["sym_visibility"].value == "private"
                ):
                    external_funcs.append(func)
                elif func.attributes["sym_name"].value == top_function_name:
                    top_func = func
                else:
                    func_name_w_id = func.attributes["sym_name"].value
                    func_name = re.match(r"^(.*?)_\d", func_name_w_id).group(1)
                    if func_name not in core_func_groups:
                        core_func_groups[func_name] = []
                    core_func_groups[func_name].append(func)
    return top_func, core_func_groups, external_funcs


def get_element_type(dtype_str: str) -> aie_ir.Type:
    """
    Convert a string representing a data type into the corresponding AIE IR type.
    """
    if dtype_str == "i32":
        return aie_ir.IntegerType.get_signless(32)
    if dtype_str == "i16":
        return aie_ir.IntegerType.get_signless(16)
    if dtype_str == "i8":
        return aie_ir.IntegerType.get_signless(8)
    if dtype_str == "f32":
        return aie_ir.F32Type.get()
    if dtype_str == "f16":
        return aie_ir.F16Type.get()
    if dtype_str == "bf16":
        return aie_ir.BF16Type.get()
    raise ValueError(f"Unsupported dtype: {dtype_str}")


def codegen_external_kernels(
    injected_kernels: dict[str, ExternalModuleBase],
    include_src: set[str],
    lib_dir: str = "aie2",
) -> str:
    """
    Generate the C++ code for external kernels to be used by the AIE compute cores.
    """
    code = """
// External kernels generated by Allo
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>
#include <aie_api/aie.hpp>
"""
    # [NOTE]: include too much may lead to 'Overflow of program memory'
    kernel_file_code = ""
    for src in include_src:
        if "mm.cc" in src:  # this file is too large to be included
            with open(
                os.path.expandvars(f"$MLIR_AIE_EXTERNAL_KERNEL_DIR/{lib_dir}/mm.cc"),
                "r",
                encoding="utf-8",
            ) as f:
                mm_kernel = f.read()
                pattern = r'#include\s+"zero\.cc"'
                mm_kernel = re.sub(pattern, f'#include "{lib_dir}/zero.cc"', mm_kernel)
                kernel_file_code += mm_kernel
        else:
            code += src

    kernel_code = ""
    for kernel in injected_kernels.values():
        code += kernel.kernel_header
        kernel_code += kernel.kernel_code

    code += '\nextern "C" {\n\n'
    code += kernel_code
    code += '} // extern "C"\n\n'

    code += kernel_file_code
    return code


# ############################################################
# Optimization Passes
# ############################################################
def collect_lib_func_call(root, kernel_name: str) -> list[allo_func_d.CallOp]:
    ops: list[allo_func_d.CallOp] = []

    def collect_recursive(op):
        if isinstance(op, allo_func_d.CallOp):
            if "lib" in op.attributes and kernel_name in op.attributes["lib"].value:
                ops.append(op.operation)
            return

        for region in op.regions:
            for block in region.blocks:
                for inner_op in block.operations:
                    collect_recursive(inner_op)

    collect_recursive(root)
    return ops


def simplify_matmul_accumulate(function: allo_func_d.FuncOp):
    """
    %alloc_0 = memref.alloc() : memref<32x32xi16>
    linalg.fill {op_name = "matmul_init_zero_0"} ins(%c0_i16 : i16) outs(%alloc_0 : memref<32x32xi16>)
    call @matmul_scalar_i16_i16(%arg0, %arg1, %alloc_0) : (memref<32x32xi16>, memref<32x32xi16>, memref<32x32xi16>) -> ()
    %alloc_1 = memref.alloc() : memref<32x32xi16>
    call @add_i16_vector(%alloc_0, %alloc, %alloc_1) : (memref<32x32xi16>, memref<32x32xi16>, memref<32x32xi16>) -> ()

    ==> (if %alloc can be write safely)
    call @matmul_scalar_i16_i16(%arg0, %arg1, %alloc) : (memref<32x32xi16>, memref<32x32xi16>, memref<32x32xi16>) -> ()
    """
    matmul_ops: list[allo_func_d.CallOp] = collect_lib_func_call(function, "matmul")
    for call_matmul_op in matmul_ops:
        output = call_matmul_op.operands[-1]
        uses = list(output.uses)
        if (
            isinstance(output.owner, allo_ir.ir.Operation)
            and output.owner.name == "memref.alloc"
            and len(uses) == 3
        ):
            init_zero_op, acc_op = allo_d.get_first_use_in_function(
                output, function
            ), allo_d.get_last_use_in_function(output, function)
            if (
                "lib" not in init_zero_op.attributes
                or "fill_zeros" not in init_zero_op.attributes["lib"].value
            ):
                init_zero_op = None
            if (
                "lib" not in acc_op.attributes
                or "add" not in acc_op.attributes["lib"].value
            ):
                acc_op = None
            if init_zero_op is not None and acc_op is not None:
                acc_base = (
                    acc_op.operands[0]
                    if acc_op.operands[0] != output
                    else acc_op.operands[1]
                )
                if (
                    allo_d.get_last_use_in_function(acc_base, function) == acc_op
                    and isinstance(acc_op.operands[-1].owner, allo_ir.ir.Operation)
                    and acc_op.operands[-1].owner.name == "memref.alloc"
                ):
                    # accumulation is the last use
                    call_matmul_op.operands[-1] = acc_base
                    init_zero_op.erase()
                    acc_op.operands[-1].replace_all_uses_with(acc_base)
                    acc_op.erase()


# ############################################################
# Run-time Utils
# ############################################################
np_supported_types = {
    "bf16": np.float32,  # numpy does not support bf16
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


def read_tensor_from_file(dtype, shape, file_path):
    arr = np.fromfile(file_path, sep="\n", dtype=np_supported_types[str(dtype)])
    return arr.reshape(shape)


# ############################################################
# Host Code Generation
# ############################################################
host_header = """
//=============================================================================
// Auto generated by Allo
//=============================================================================
#include <boost/program_options.hpp>
#include <bits/stdc++.h>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <stdfloat>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "test_utils.h"

namespace po = boost::program_options;

int main(int argc, const char *argv[]) {
  // ------------------------------------------------------
  // Parse program arguments
  // ------------------------------------------------------
  po::options_description options("Allowed options");
  po::variables_map vm;
  test_utils::add_default_options(options);
  options.add_options()
    ("profile,p", po::value<bool>()->default_value(false), "enable profiling")
    ("test_iter,t", po::value<int>()->default_value(100), "number of test iterations");

  test_utils::parse_options(argc, argv, options, vm);
  int verbosity = vm["verbosity"].as<int>();
  int do_verify = vm["verify"].as<bool>();
  int n_iterations = vm["iters"].as<int>();
  int n_warmup_iterations = vm["warmup"].as<int>();
  int trace_size = vm["trace_sz"].as<int>();
  bool do_profile = vm["profile"].as<bool>();
  int n_test_iterations = vm["test_iter"].as<int>();

  // Load instruction sequence
  std::vector<uint32_t> instr_v =
      test_utils::load_instr_binary(vm["instr"].as<std::string>());
  if (verbosity >= 1)
    std::cout << "Sequence instr count: " << instr_v.size() << std::endl;

  // ------------------------------------------------------
  // Get device, load the xclbin & kernel and register them
  // ------------------------------------------------------
  unsigned int device_index = 0;
  auto device = xrt::device(device_index);

  // Load the xclbin
  if (verbosity >= 1)
    std::cout << "Loading xclbin: " << vm["xclbin"].as<std::string>() << std::endl;
  auto xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());

  // Load the kernel
  if (verbosity >= 1)
    std::cout << "Kernel opcode: " << vm["kernel"].as<std::string>() << std::endl;
  std::string Node = vm["kernel"].as<std::string>();
  // Get the kernel from the xclbin
  auto xkernels = xclbin.get_kernels();
  auto xkernel = *std::find_if(xkernels.begin(), xkernels.end(),
                               [Node, verbosity](xrt::xclbin::kernel &k) {
                                 auto name = k.get_name();
                                 if (verbosity >= 1) {
                                   std::cout << "Name: " << name << std::endl;
                                 }
                                 return name.rfind(Node, 0) == 0;
                               });
  auto kernelName = xkernel.get_name();

  // Register xclbin
  if (verbosity >= 1)
    std::cout << "Registering xclbin: " << vm["xclbin"].as<std::string>() << std::endl;
  device.register_xclbin(xclbin);

  // Get a hardware context
  if (verbosity >= 1)
    std::cout << "Getting hardware context." << std::endl;
  xrt::hw_context context(device, xclbin.get_uuid());

  // Get a kernel handle
  if (verbosity >= 1)
    std::cout << "Getting handle to kernel:" << kernelName << std::endl;
  auto kernel = xrt::kernel(context, kernelName);

  // instruction
  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int), XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));
  // output
  std::ofstream ofile("output.data");
  if (!ofile.is_open()) {
      std::cerr << "Error: Could not open output file.\\n";
      return 1;
  }

  // kernel arguments
  unsigned int opcode = 3;
"""

file_close_str = """  ofile.close();
  if (verbosity >= 1)
    std::cout << "Array has been written to output.data.\\n";
  return 0;
}
"""


def codegen_host(inputs: dict[int, DTensor], outputs: dict[int, DTensor]):
    """
    Generate the C++ code for external kernels for host CPU.
    """
    code = host_header
    with format_code(indent=2):
        # write input data
        for i in range(len(inputs)):
            dtensor = inputs[i]
            shape = dtensor.shape
            dtype = aie_ctype_map[str(dtensor.dtype)]
            code += format_str(f'std::ifstream ifile{i}("input{i}.data");')
            code += format_str(f"if (!ifile{i}.is_open()) {{")
            code += format_str(
                '  std::cerr << "Error: Could not open input file.\\n";', strip=False
            )
            code += format_str("  return 1;", strip=False)
            code += format_str("}")
            size = np.prod(shape)
            code += format_str(
                f"auto bo_in{i} = xrt::bo(device, {size} * sizeof({dtype}),"
            )
            with format_code(indent=24):
                code += format_str(
                    f"XRT_BO_FLAGS_HOST_ONLY, kernel.group_id({i + 3}));"
                )
            code += format_str(f"{dtype} *bufIn{i} = bo_in{i}.map<{dtype} *>();")
            code += format_str(f"std::vector<{dtype}> srcVec{i};")
            code += format_str(f"for (int i = 0; i < {size}; i++) {{")
            with format_code(indent=4):
                code += format_str(f"{dtype} num;")
                code += format_str(f"ifile{i} >> num;")
                code += format_str(f"srcVec{i}.push_back(num);")
            code += format_str("}")
            code += format_str(
                f"memcpy(bufIn{i}, srcVec{i}.data(), (srcVec{i}.size() * sizeof({dtype})));"
            )
        for i in range(len(outputs)):
            dtensor = outputs[i + len(inputs)]
            shape = dtensor.shape
            dtype = aie_ctype_map[str(dtensor.dtype)]
            out_size = np.prod(shape)
            code += format_str(
                f"\nauto bo_out{i} = xrt::bo(device, {out_size} * sizeof({dtype}),",
                strip=False,
            )
            with format_code(indent=24):
                code += format_str(
                    f"XRT_BO_FLAGS_HOST_ONLY, kernel.group_id({len(inputs) + 2 + i}));"
                )
        # trace
        code += format_str(
            "int tmp_trace_size = (trace_size > 0) ? trace_size : 1;", strip=False
        )
        code += format_str(
            f"auto bo_trace = xrt::bo(device, tmp_trace_size * 4, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id({len(inputs) + len(outputs) + 3}));"
        )
        code += format_str("if (verbosity >= 1)")
        code += format_str(
            '  std::cout << "Writing data into buffer objects.\\n";', strip=False
        )
        code += format_str("char *bufTrace = bo_trace.map<char *>();")
        code += format_str("if (trace_size > 0)")
        code += format_str("  memset(bufTrace, 0, trace_size);", strip=False)

        code += format_str("\nbo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);", strip=False)
        for i in range(len(inputs)):
            code += format_str(f"bo_in{i}.sync(XCL_BO_SYNC_BO_TO_DEVICE);")
        code += format_str("if (trace_size > 0)")
        code += format_str("  bo_trace.sync(XCL_BO_SYNC_BO_TO_DEVICE);", strip=False)
        # run kernels
        code += format_str("if (verbosity >= 1)")
        code += format_str('  std::cout << "Running Kernel.\\n";', strip=False)
        inbufs = ", ".join([f"bo_in{i}" for i in range(len(inputs))])
        outbufs = ", ".join([f"bo_out{i}" for i in range(len(outputs))])
        code += format_str("if (!do_profile) {")
        with format_code(indent=4):
            code += format_str(
                "auto start = std::chrono::high_resolution_clock::now();", strip=False
            )
            code += format_str("// gid: (opcode, instr, instr_size, ...)")
            code += format_str(
                f"auto run = kernel(opcode, bo_instr, instr_v.size(), {inbufs}, {outbufs}, bo_trace);"
            )
            code += format_str("ert_cmd_state r = run.wait();")
            code += format_str(
                "\nauto end = std::chrono::high_resolution_clock::now();", strip=False
            )
            code += format_str("if (r != ERT_CMD_STATE_COMPLETED) {")
            with format_code(indent=8):
                code += format_str(
                    'std::cout << "Kernel did not complete. Returned status: " << r << "\\n";'
                )
                code += format_str("return 1;")
            code += format_str("}")
            code += format_str(
                "float npu_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();"
            )
            code += format_str(
                'std::cout << "NPU execution time: " << npu_time << "us\\n";'
            )
        code += format_str("} else {")
        with format_code(indent=4):
            code += format_str("for (size_t i = 0; i < n_warmup_iterations; i++) {")
            with format_code(indent=8):
                code += format_str(
                    f"auto run = kernel(opcode, bo_instr, instr_v.size(), {inbufs}, {outbufs}, bo_trace);"
                )
                code += format_str("ert_cmd_state r = run.wait();")
                code += format_str("if (r != ERT_CMD_STATE_COMPLETED) {")
                with format_code(indent=12):
                    code += format_str(
                        'std::cout << "Kernel did not complete. Returned status: " << r << "\\n";'
                    )
                    code += format_str("return 1;")
                code += format_str("}")
            code += format_str("}")
            code += format_str("float total_npu_time = 0;")
            code += format_str("for (size_t i = 0; i < n_test_iterations; i++) {")
            with format_code(indent=8):
                code += format_str(
                    "auto start = std::chrono::high_resolution_clock::now();",
                    strip=False,
                )
                code += format_str(
                    f"auto run = kernel(opcode, bo_instr, instr_v.size(), {inbufs}, {outbufs});"
                )
                code += format_str("run.wait();")
                code += format_str(
                    "\nauto end = std::chrono::high_resolution_clock::now();",
                    strip=False,
                )
                code += format_str(
                    "float npu_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();"
                )
                code += format_str("total_npu_time += npu_time;")
            code += format_str("}")
            code += format_str(
                'std::cout << "Avg NPU execution time: " << total_npu_time / n_test_iterations << "us\\n";'
            )
        code += format_str("}")
        # get results
        for i in range(len(outputs)):
            dtensor = outputs[i + len(inputs)]
            shape = dtensor.shape
            dtype = aie_ctype_map[str(dtensor.dtype)]
            out_size = np.prod(shape)
            code += format_str(
                f"\nbo_out{i}.sync(XCL_BO_SYNC_BO_FROM_DEVICE);", strip=False
            )
            code += format_str(f"{dtype} *bufOut{i} = bo_out{i}.map<{dtype} *>();")
            code += format_str(f"for (uint32_t i = 0; i < {out_size}; i++) {{")
            code += format_str(f'  ofile << *(bufOut{i} + i) << "\\n";', strip=False)
            code += format_str("}")
        code += format_str("\n// Close files", strip=False)
        for i in range(len(inputs)):
            code += format_str(f"ifile{i}.close();")
        code += format_str("if (trace_size > 0) {")
        code += format_str("  bo_trace.sync(XCL_BO_SYNC_BO_FROM_DEVICE);", strip=False)
        code += format_str(
            '  test_utils::write_out_trace(((char *)bufTrace), trace_size, vm["trace_file"].as<std::string>());',
            strip=False,
        )
        code += format_str("}")
        code += file_close_str

    return code


# ############################################################
# Tools
# ############################################################


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        # Path compression
        if self.parent.setdefault(x, x) != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        self.parent[self.find(x)] = self.find(y)


def merge_token_sets(token_sets: list) -> list:
    uf = UnionFind()
    # union all overlapping tokens
    for token_set in token_sets:
        token_list = list(token_set)
        for i in range(1, len(token_list)):
            uf.union(token_list[0], token_list[i])
    # group tokens
    groups = {}
    for token_set in token_sets:
        for token in token_set:
            root = uf.find(token)
            groups.setdefault(root, set()).add(token)
    return list(groups.values())


def string_sort_key(s: str):
    nums = tuple(int(x) for x in re.findall(r"\d+", s))
    return (len(nums), nums)
