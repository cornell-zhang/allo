# pylint: disable=import-error, no-name-in-module, c-extension-no-member, too-many-branches, too-many-nested-blocks, redefined-variable-type
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from collections import defaultdict
from dataclasses import dataclass
import numpy as np

# =======================

import aie.dialects.aie as aie_d
import aie.dialects.aiex as aiex_d
import aie.dialects.arith as aie_arith_d
import aie.dialects.func as aie_func_d
import aie.dialects.scf as aie_scf_d

import aie.ir as aie_ir

# =======================

import allo._mlir._mlir_libs._mlir as allo_ir
import allo._mlir.dialects._memref_ops_gen as allo_memref_d

from ..._mlir.ir import (
    MemRefType,
    InsertionPoint,
    Context,
    Type,
    IntegerType,
    F16Type,
    F32Type,
    BF16Type,
)

from ..utils import format_str
from ..._mlir.dialects import func as allo_func_d
from ...memory import DTensor

from .utils import get_element_type
from ..aie import map_kernels_to_device_mesh


class Stream:
    """
    Allo Stream class
    """

    def __init__(self, name: str):
        self.name = name
        self.type_str = None
        self.depth = -1
        self.shape: list[int] = None
        self.dtype: str = None
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
        if self.depth >= 0:
            assert type_str == self.type_str
            return
        match = re.match(r"!allo\.stream<([^,]+),\s*(\d+)>", type_str)
        shape: list[int] = None
        dtype: str = None
        if match:
            with context, allo_ir.ir.Location.unknown():
                element_type_str = match.group(1)
                self.depth = int(match.group(2))
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
                self.shape = shape
                self.dtype = dtype
                self.is_tensor = len(shape) > 0
        else:
            raise ValueError(f"Invalid stream type {type_str}.")

    def __str__(self):
        return f"Stream (name={self.name}, depth={self.depth}, dtype={self.allo_element_type}, is_tensor={self.is_tensor}, src={self.src}, dst={self.dst})"


@dataclass
class Argument:
    """
    Represents an argument to a function, either a DTensor or a Stream.
    """

    dtensor: DTensor
    stream: Stream


@dataclass(frozen=True)
class DMATensorTile:
    dtensor_tile_id: int  # dTensor may need to be further partitioned
    shim_id: int
    mem_id: int
    tensor_tile_labels: list
    offset: list
    size: list
    stride: list


def map_global_io(inputs, outputs) -> tuple[dict[str, list[DMATensorTile]], int, int]:
    """
    TODO: make use of the fifth mem tile
    TODO: The current mapping scheme requires the matrices to be completely partitioned without remaining elements (shape should be divided by tile num).

    Allocate (shim-tile, mem-tile) pairs for every DTensor that crosses the
    NPU boundary, while respecting the per-mem-tile ObjectFIFO limits.

    The algorithm tries to pack as much traffic as possible into the fewest
    number of memory tiles, only falling back to splitting (called "parts")
    when the requested number of FIFOs would exceed the quota per memory tile.

    Current constrains:
        - use 4 mem-shim tile pairs for io
        - each port is assigned to one dtensor tile

    Args:
        - inputs: A dictionary mapping function names (group name + id) to lists of objects as inputs.
        - outputs: A dictionary mapping function names (group name + id) to lists of objects as outputs.
    Return:
        - tile_map: dtensor name -> a list of dma tiles
        - mem_tile_num
        - shim_tile_num
    """
    MAX_MEM_TILES = 4  # Maximum number of memory tiles allowed
    # https://riallto.ai/notebooks/3_2_Ryzenai_capabilities.html#memory-tile-properties
    MAX_SEND = 6  # Maximum number of producer FIFOs per memory tile (DMA limits)
    MAX_RECV = 6  # Maximum number of consumer FIFOs per memory tile (DMA limits)

    @dataclass
    class Tile:
        send_number: int
        recv_number: int

    used_tiles: list[Tile] = []

    def assign_tile(send_need, recv_need) -> int:
        """
        Try to assign a memory tile satisfying the requirement.
        Return the tile index.
            -1 indicates no tile availability.
        """
        # 1. Attempt to use a new memory tile
        if (
            len(used_tiles) < MAX_MEM_TILES
            and send_need <= MAX_SEND
            and recv_need <= MAX_RECV
        ):
            used_tiles.append(Tile(send_need, recv_need))
            return len(used_tiles) - 1
        # 2. Otherwise, try to pack into an existing tile
        for i, _ in enumerate(used_tiles):
            if (
                used_tiles[i].send_number + send_need <= MAX_SEND
                and used_tiles[i].recv_number + recv_need <= MAX_RECV
            ):
                used_tiles[i].send_number += send_need
                used_tiles[i].recv_number += recv_need
                return i
        # 3. No tile fits
        return -1

    def map_dtensor_to_tile(dtensor: DTensor, is_input: bool):
        """
        Split a DTensor into Part instances so each Part fits on some memory tile with respect to FIFO limits.

        Currently, we focus on dtensor io using memory tiles.
        Shim tiles are assigned using a one-to-one mapping from memory tiles.

        DTensors are sent to or from compute cores.
        Since memory tile is used for transfer, we assume that `receive` implies one `send` and `send` implies one `receive`.
        """
        device_dims, size, stride = dtensor.get_access_pattern()
        tensor_tiles = sorted(
            list(dtensor.global_placement.keys())
        )  # 'R' can use one port yet multiple destinations
        send_need = len(tensor_tiles) if is_input else 1
        recv_need = 1 if is_input else len(tensor_tiles)
        mem_tile_id = assign_tile(send_need, recv_need)
        if mem_tile_id >= 0:
            return [
                DMATensorTile(
                    0,
                    mem_tile_id,
                    mem_tile_id,
                    tensor_tiles,
                    [0, 0, 0, 0],
                    size,
                    stride,
                )
            ]
        # We failed to transfer the whole tensor with one memory tile. Try using more.
        dma_tensor_tiles: list[DMATensorTile] = []
        # fixme: incomplete
        #   Currently, we may allow tensor tiles on a sharding dimension to be sent using different memory tiles
        assert len(device_dims) == 1 or len(device_dims) == 2
        lose_factor = 1 if len(device_dims) <= 1 else size[device_dims[0]]
        inc_factor = (
            size[device_dims[0]] if len(device_dims) <= 1 else size[device_dims[1]]
        )
        remaining = tensor_tiles[:]
        start_idx = 0
        while remaining:
            offset = [0, 0, 0, 0]
            chunk = remaining
            while chunk:
                send_need = len(chunk) if is_input else 1
                recv_need = 1 if is_input else len(chunk)
                mem_tile_id = assign_tile(send_need, recv_need)
                if mem_tile_id >= 0:
                    break
                chunk = chunk[: (len(chunk) - lose_factor)]  # Reduce size and retry
            if not chunk:
                raise RuntimeError(
                    "Failed to allocate (shim, memory) tile: per-tile FIFO limit "
                    "exceeded or no more available tiles."
                )
            offset[device_dims[0]] = start_idx // inc_factor
            size[device_dims[0]] = len(chunk) // inc_factor
            dma_tensor_tiles.append(
                DMATensorTile(
                    len(dma_tensor_tiles),
                    mem_tile_id,
                    mem_tile_id,
                    chunk,
                    offset,
                    size,
                    stride,
                )
            )
            remaining = remaining[len(chunk) :]
            start_idx += len(chunk)
        return dma_tensor_tiles

    tile_map: dict[str, list[DMATensorTile]] = defaultdict(list)

    for io_lst, is_input in ((inputs, True), (outputs, False)):
        for _, sub in io_lst.items():
            for dtensor in sub["_global"]:
                tile_map[dtensor.name].extend(
                    map_dtensor_to_tile(dtensor, is_input=is_input)
                )

    return tile_map, len(used_tiles), len(used_tiles)


class CodeGenerator:
    """
    CodeGenerator is responsible for transforming Allo functions and their associated
    DTensor-based input/output mappings into AIE (AI Engine) core-compatible IR.
    It manages stream transformations, memory operations, and integrates with the
    AIE dialect of MLIR.
    """

    def __init__(
        self,
        device_type: str,
        global_inputs: dict[int, DTensor],
        global_outputs: dict[int, DTensor],
        top_function: allo_func_d.FuncOp,
    ):
        self.device_type = device_type

        self.global_inputs: dict[int, DTensor] = global_inputs
        self.global_outputs: dict[int, DTensor] = global_outputs
        self.top_function = top_function

        self.tile_map: dict[str, aie_d.TileOp] = {}
        self.fifo_map: dict[str, aie_d.object_fifo] = {}
        # function name (with id) -> a map from DTensor to fifo name
        self.compute_core_io: dict[str : dict[DTensor, str]] = {}
        self.external_functions: str = ""

        self.aie_module = None  # The top-level AIE IR module
        self.global_ip: aie_ir.InsertionPoint = (
            None  # mark the inserting point for buffers
        )

    def collect_stream_info(self, streams: dict[str, Stream], context):
        """
        Extract and update allo.stream element types from the top_function body.

        Args:
            - streams (dict[str, Stream]): Dictionary of allo.stream objects.
            - context: The current allo MLIR context.
        """
        for func_block in self.top_function.body:
            for op in func_block.operations:
                if op.name == "allo.stream_construct":
                    streams[op.attributes["name"].value].set_element_type(
                        str(op.res.type), context
                    )

    def preporocess_dumped_core_func(
        self,
        original_func: allo_func_d.FuncOp,
        func_args: dict[int, tuple[Argument, bool]],
    ) -> str:
        """
        Preprocess the core function in allo MLIR.

        Args:
            - original_func (FuncOp): The function in allo MLIR to transform.
            - func_args (dict): Maps function argument indices to (Argument, is_output) pairs.

        Returns:
            - str: A string representation of the rewritten function with allo.stream ops replaced.
        """
        # replace pipe with memref operations
        with original_func.context, allo_ir.ir.Location.unknown():
            func_inputs = original_func.type.inputs
            for idx, arg_info in func_args.items():
                if arg_info[0].stream is not None:
                    func_inputs[idx] = arg_info[0].stream.allo_element_type
            func_type = allo_func_d.FunctionType.get(
                func_inputs,
                original_func.type.results,
            )
            new_function = allo_func_d.FuncOp(
                original_func.name.value,
                func_type,
                ip=InsertionPoint(original_func),
            )
            entry_block = new_function.add_entry_block()
            for old, new in zip(original_func.arguments, new_function.arguments):
                old.replace_all_uses_with(new)
            with InsertionPoint(entry_block):
                for func_block in original_func.body:
                    for op in func_block.operations:
                        new_op = op.clone()
                        for old, new in zip(op.results, new_op.results):
                            old.replace_all_uses_with(new)
            original_func.erase()
            for idx, arg_info in func_args.items():
                if arg_info[0].stream is not None:
                    argument = new_function.arguments[idx]
                    for use_ in argument.uses:
                        op = use_.owner
                        if op.name == "allo.stream_put":
                            operands = op.operands
                            # store/copy
                            if arg_info[0].stream.is_tensor:
                                new_op = allo_memref_d.CopyOp(
                                    operands[1], operands[0], ip=InsertionPoint(op)
                                )
                            else:
                                new_op = allo_memref_d.StoreOp(
                                    operands[1], operands[0], [], ip=InsertionPoint(op)
                                )
                        elif op.name == "allo.stream_get":
                            # load/alloc
                            if arg_info[0].stream.is_tensor:
                                # replace use with alloc
                                new_op = allo_memref_d.AllocOp(
                                    arg_info[0].stream.allo_element_type,
                                    [],
                                    [],
                                    ip=InsertionPoint(op),
                                )
                                # use copy to track
                                allo_memref_d.CopyOp(
                                    op.operands[0], new_op.memref, ip=InsertionPoint(op)
                                )
                            else:
                                new_op = allo_memref_d.LoadOp(
                                    argument, [], ip=InsertionPoint(op)
                                )
                        else:
                            continue
                        # replace use
                        for old, new in zip(op.results, new_op.results):
                            old.replace_all_uses_with(new)
                        op.erase()

        # declare external kernel function before use
        func_str = self.external_functions + "\n" + str(new_function)
        return func_str

    def build_core_function(
        self,
        func_core: aie_d.Core,
        original_func: allo_func_d.FuncOp,
        func_args: dict[int, tuple[Argument, bool]],
    ):
        """
        Generate the computation logic for the fake 'while(1)' loop body for an AIE compute core, transforming high-level Allo ops
        into AIE MLIR.

        Args:
            - func_core (aie_d.Core): The target compute core to insert into.
            - original_func (FuncOp): The Allo function to compile.
            - func_args (dict): Maps argument indices to (Argument, is_output) tuples.
        """
        func_string = self.preporocess_dumped_core_func(original_func, func_args)
        original_module = aie_ir.Module.parse(func_string)
        parsed_function: aie_func_d.FuncOp = None
        for func in original_module.body.operations:
            if isinstance(func, aie_func_d.FuncOp):
                if not (
                    "sym_visibility" in func.attributes
                    and func.attributes["sym_visibility"].value == "private"
                ):
                    if parsed_function is None:
                        parsed_function = func
                    else:
                        raise ValueError("Too many core functions. Fail to resolve.")
        assert not parsed_function is None
        func_body = func_core.regions[0]
        entry_block = aie_ir.Block.create_at_start(func_body)
        with aie_ir.InsertionPoint(entry_block):
            index_type = aie_ir.IndexType.get()
            # compute core wrapper: fake while(1)
            c0 = aie_arith_d.ConstantOp(value=0, result=index_type)
            c1 = aie_arith_d.ConstantOp(value=1, result=index_type)
            cmax = aie_arith_d.ConstantOp(value=9223372036854775807, result=index_type)
            # scf.for %arg0 = %c0 to %cmax step %c1
            loop = aie_scf_d.ForOp(lower_bound=c0, upper_bound=cmax, step=c1)
            with aie_ir.InsertionPoint(loop.body):
                # insert operations to get 'function parameter', acquire and subview
                io_map = (
                    self.compute_core_io[parsed_function.name.value]
                    if parsed_function.name.value in self.compute_core_io
                    else {}
                )
                for i, argument in enumerate(parsed_function.arguments):
                    if not i in func_args:
                        continue
                    arg_info: tuple[Argument, bool] = func_args[i]
                    if arg_info[0].dtensor is not None:
                        acquired = self.fifo_map[io_map[arg_info[0].dtensor]].acquire(
                            1 if arg_info[1] else 0, 1
                        )
                        argument.replace_all_uses_with(acquired)
                    else:
                        stream: Stream = arg_info[0].stream
                        fifo = self.fifo_map[stream.name]
                        for use_ in argument.uses:
                            op = use_.owner
                            with aie_ir.InsertionPoint(op.operation):
                                if op.name == "memref.store" or (
                                    op.name == "memref.copy"
                                    and argument == op.operands[1]
                                ):  # allo.stream_put
                                    acquired = fifo.acquire(0, 1)
                                    op.operands[1] = acquired
                                    new_op = op.clone()  # no use, no need to replace
                                    fifo.release(0, 1)
                                    op.erase()
                                elif (
                                    op.name == "memref.load"
                                ):  # allo.stream_get, non-tensor
                                    acquired = fifo.acquire(1, 1)
                                    op.operands[0] = acquired
                                    new_op = op.clone()
                                    for old, new in zip(op.results, new_op.results):
                                        old.replace_all_uses_with(new)
                                    fifo.release(1, 1)
                                    op.erase()
                                elif (
                                    op.name == "memref.copy"
                                ):  # allo.stream_get, tensor
                                    acquired = fifo.acquire(1, 1)
                                    op.operands[0] = acquired
                                    new_op = op.clone()
                                    fifo.release(1, 1)
                                    op.erase()

                for parsed_func_block in parsed_function.body:
                    for op in parsed_func_block.operations:
                        if op.name == "func.return":
                            continue
                        new_op = op.clone()
                        for old, new in zip(op.results, new_op.results):
                            old.replace_all_uses_with(new)

                # replace alloc with buffer
                alloc_ops = []

                def collect_allocs(op):
                    if op.name == "memref.alloc":
                        alloc_ops.append(op.operation)
                        return
                    for region in op.regions:
                        for block in region.blocks:
                            for inner_op in block.operations:
                                collect_allocs(inner_op)

                collect_allocs(loop)
                for alloc_op in alloc_ops:
                    buffer_op = aie_d.BufferOp(
                        buffer=alloc_op.results[0].type,
                        tile=self.tile_map[f"compute_{parsed_function.name.value}"],
                        ip=self.global_ip,
                    )
                    for old, new in zip(alloc_op.results, buffer_op.results):
                        old.replace_all_uses_with(new)
                    alloc_op.erase()

                # release
                for i, _ in enumerate(parsed_function.arguments):
                    if not i in func_args:
                        continue
                    arg_info: tuple[Argument, bool] = func_args[i]
                    if not arg_info[0].dtensor is None:
                        self.fifo_map[io_map[arg_info[0].dtensor]].release(
                            1 if arg_info[1] else 0, 1
                        )

                aie_scf_d.YieldOp([])
            aie_d.EndOp()

    def aie_codegen(
        self,
        core_func_groups: dict[str, list[allo_func_d.FuncOp]],
        external_funcs: list[allo_func_d.FuncOp],
        inputs,
        outputs,
        use_external_kernels: dict[str, bool],
        core_func_args: dict[str, dict[int, tuple[Argument, bool]]],
        streams: dict[str, Stream],
    ) -> aie_ir.Module:
        """
        Generate an AIE MLIR module.
        """
        io_mapping, mem_tile_num, shim_tile_num = map_global_io(inputs, outputs)
        wrapper_code = f"""
            module {{
                aie.device({self.device_type}) {{
        """

        # fixme: maybe better to resolve this using IR constructor
        for func in external_funcs:
            self.external_functions += format_str(str(func), indent=4)

        wrapper_code += self.external_functions
        wrapper_code += """
                }
            }
        """
        with aie_ir.Context() as ctx, aie_ir.Location.unknown():
            # module wrapper
            self.aie_module = aie_ir.Module.parse(wrapper_code, ctx)
            # find device op: aie.device(device_type)
            device_op = None
            for op in self.aie_module.body.operations:
                if isinstance(op, aie_d.DeviceOp):
                    device_op = op
                    break
            assert device_op is not None, "aie.device not found"
            device_body = device_op.regions[0].blocks[0]
            # insert operations in the device body, before `aie.end``
            end_op = None
            for op in device_body.operations:
                if isinstance(op, aie_d.EndOp):
                    end_op = op
                    break
            assert not end_op is None

            with aie_ir.InsertionPoint(end_op):
                # shim tile
                for shim_id in range(shim_tile_num):
                    self.tile_map[f"shim_{shim_id}"] = aie_d.TileOp(col=shim_id, row=0)
                # mem tiles
                for mem_id in range(mem_tile_num):
                    self.tile_map[f"mem_{mem_id}"] = aie_d.TileOp(col=mem_id, row=1)
                # compute tiles
                # 'logic' mapping for different function groups.
                mappings = {}
                for func_name in core_func_groups:
                    if len(inputs[func_name]["_global"]) > 0:
                        mappings[func_name] = inputs[func_name]["_global"][0].mapping
                    else:
                        mappings[func_name] = outputs[func_name]["_global"][0].mapping
                aie_mesh = (5, 4)
                for func_name, tile_ids in map_kernels_to_device_mesh(
                    mappings, aie_mesh
                ).items():
                    for idx, func in zip(tile_ids, core_func_groups[func_name]):
                        func_name = func.attributes["sym_name"].value
                        self.tile_map[f"compute_{func_name}"] = aie_d.TileOp(
                            col=idx[0],
                            row=idx[1] + 2,
                        )

                for io, arg_lst in (("in", inputs), ("out", outputs)):
                    for func_name, sub_func_lst in arg_lst.items():
                        for idx, dtensor in enumerate(sub_func_lst["_global"]):
                            placement = dtensor.global_placement
                            # shim <-> mem (one to one)
                            for dma_tile in io_mapping[dtensor.name]:
                                # define objectfifo
                                name = f"{io}_shim_{dtensor.name}{dma_tile.dtensor_tile_id}"
                                producer = (
                                    self.tile_map[f"shim_{dma_tile.shim_id}"]
                                    if io == "in"
                                    else self.tile_map[f"mem_{dma_tile.mem_id}"]
                                )
                                consumer = (
                                    [self.tile_map[f"mem_{dma_tile.mem_id}"]]
                                    if io == "in"
                                    else [self.tile_map[f"shim_{dma_tile.shim_id}"]]
                                )
                                idx_ = next(
                                    (
                                        i
                                        for i, size in enumerate(dma_tile.size)
                                        if size != 1
                                    ),
                                    None,
                                )
                                if idx_ is None:
                                    shape = [1]
                                else:
                                    shape = dma_tile.size[idx_:]
                                memref_type = aie_ir.MemRefType.get(
                                    shape,
                                    get_element_type(str(dtensor.dtype)),
                                )
                                fifo_shim = self.fifo_map[name] = aie_d.object_fifo(
                                    name,
                                    producer,
                                    consumer,
                                    depth=2,
                                    datatype=memref_type,
                                )

                                # mem <-> compute (one to ?)
                                fifo_mem = []
                                mem_stride = [0]
                                mem_tile = self.tile_map[f"mem_{dma_tile.mem_id}"]
                                local_memref_type = aie_ir.MemRefType.get(
                                    dtensor.type_as_param,
                                    get_element_type(str(dtensor.dtype)),
                                )
                                for tensor_tile in dma_tile.tensor_tile_labels:
                                    # distribute to placement[tensor_tile]
                                    compute_tiles = []
                                    name = f"{io}_mem_{dtensor.name}_{tensor_tile}"
                                    for tile in placement[tensor_tile]:
                                        # some distributed tile do not have global output
                                        if (
                                            dtensor in outputs[func_name][tile]
                                            or dtensor in inputs[func_name][tile]
                                        ):
                                            idx_str = "_".join([str(x) for x in tile])
                                            # seems confusing. "sym_name" is parsed in this way
                                            compute_tiles.append(
                                                self.tile_map[
                                                    f"compute_{func_name}_{idx_str}"
                                                ]
                                            )
                                            self.compute_core_io.setdefault(
                                                f"{func_name}_{idx_str}", {}
                                            )[dtensor] = name
                                    if io == "in":
                                        producer = mem_tile
                                    else:
                                        # fixme: only one valid producer
                                        assert len(compute_tiles) == 1
                                        producer = compute_tiles[0]
                                    consumer = (
                                        compute_tiles if io == "in" else [mem_tile]
                                    )
                                    fifo = self.fifo_map[name] = aie_d.object_fifo(
                                        name,
                                        producer,
                                        consumer,
                                        depth=2,
                                        datatype=local_memref_type,
                                    )
                                    fifo_mem.append(fifo)
                                    mem_stride.append(
                                        mem_stride[-1]
                                        + np.prod(dtensor.get_local_shape())
                                    )
                                aie_d.object_fifo_link(
                                    fifo_shim if io == "in" else fifo_mem,
                                    fifo_mem if io == "in" else fifo_shim,
                                    [] if io == "in" else mem_stride[:-1],
                                    mem_stride[:-1] if io == "in" else [],
                                )
                # compute <-> compute
                self.collect_stream_info(streams, self.top_function.context)
                for stream_name, stream in streams.items():
                    src_tile = self.tile_map[f"compute_{stream.src}"]
                    dst_tile = [self.tile_map[f"compute_{stream.dst}"]]
                    self.fifo_map[stream_name] = aie_d.object_fifo(
                        stream_name,
                        src_tile,
                        dst_tile,
                        depth=stream.depth,
                        datatype=aie_ir.MemRefType.get(
                            stream.shape,
                            get_element_type(str(stream.dtype)),
                        ),
                    )

                for func_name in core_func_groups:
                    for func in core_func_groups[func_name]:
                        func_name_w_id = func.attributes["sym_name"].value
                        func_core = aie_d.Core(
                            tile=self.tile_map[f"compute_{func_name_w_id}"],
                            link_with=(
                                "external.o"
                                if use_external_kernels[func_name_w_id]
                                else None
                            ),
                        )
                        if self.global_ip is None:
                            self.global_ip = aie_ir.InsertionPoint(func_core)
                        self.build_core_function(
                            func_core, func, core_func_args[func_name_w_id]
                        )

                # runtime sequence
                runtime_seq = aiex_d.RuntimeSequenceOp()
                runtime_args = []
                for i in range(len(self.global_inputs) + len(self.global_outputs)):
                    arg = (
                        self.global_inputs[i]
                        if i in self.global_inputs
                        else self.global_outputs[i]
                    )
                    runtime_args.append(
                        aie_ir.MemRefType.get(
                            arg.shape, get_element_type(str(arg.dtype))
                        )
                    )

                runtime_seq_entry_block = runtime_seq.body.blocks.append(*runtime_args)
                with aie_ir.InsertionPoint(runtime_seq_entry_block):
                    dma_tiles: list = []
                    bd_cnt = 0
                    for i in range(len(self.global_inputs) + len(self.global_outputs)):
                        io = "in" if i in self.global_inputs else "out"
                        dtensor = (
                            self.global_inputs[i]
                            if i in self.global_inputs
                            else self.global_outputs[i]
                        )

                        for dma_tile in io_mapping[dtensor.name]:
                            dma_fifo = self.fifo_map[
                                f"{io}_shim_{dtensor.name}{dma_tile.dtensor_tile_id}"
                            ]
                            aiex_d.NpuDmaMemcpyNd(
                                metadata=dma_fifo,
                                bd_id=bd_cnt,
                                mem=runtime_seq_entry_block.arguments[i],
                                offsets=dma_tile.offset,
                                sizes=dma_tile.size,
                                strides=dma_tile.stride,
                                issue_token=True,
                            )
                            bd_cnt += 1
                            dma_tiles.append(dma_fifo)
                    # DMA wait
                    for dma_tile in dma_tiles:
                        aiex_d.dma_wait(dma_tile)
                    aie_d.EndOp()

        return self.aie_module
