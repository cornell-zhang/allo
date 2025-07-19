# pylint: disable=import-error, no-name-in-module, c-extension-no-member, too-many-branches, too-many-nested-blocks, redefined-variable-type, consider-using-enumerate, too-many-instance-attributes, chained-comparison
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
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
from allo._mlir.dialects import (
    func as allo_func_d,
    _memref_ops_gen as allo_memref_d,
)

from ..._mlir.ir import InsertionPoint, MemRefType, IntegerType

from ..utils import format_str
from ...memory import (
    DTensor,
    Offset4D,
    Size4D,
    coalesce_memory_access,
)

from .utils import (
    get_element_type,
    collect_op_by_name,
    merge_token_sets,
    device_config_map,
    Argument,
    Stream,
    Config,
    string_sort_key,
)
from ..ai_engine import map_kernels_to_device_mesh
from .mapping import (
    SwitchNode,
    PEInterface,
    LiveDTensorTileGroup,
    DTensorTileGroup,
    ComputationGraph,
    FIFO,
    FIFOManager,
)


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
            and send_need <= Config.MEM_MAX_SEND
            and recv_need <= Config.MEM_MAX_RECV
        ):
            used_tiles.append(Tile(send_need, recv_need))
            return len(used_tiles) - 1
        # 2. Otherwise, try to pack into an existing tile
        for i, _ in enumerate(used_tiles):
            if (
                used_tiles[i].send_number + send_need <= Config.MEM_MAX_SEND
                and used_tiles[i].recv_number + recv_need <= Config.MEM_MAX_RECV
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
        assert dtensor.access_pattern_set, "Access pattern is not set for dtensor"
        device_dims, size, stride = dtensor.shared_dims, dtensor.size, dtensor.stride
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
        if len(device_dims) <= 1:
            lose_factor, inc_factor = 1, 1
        elif len(device_dims) == 2:
            lose_factor = size[device_dims[0]]
            inc_factor = size[device_dims[1]]
        else:
            raise ValueError("Unsupported access pattern.")
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
        core_func_args: dict[str, dict[int, tuple[Argument, bool]]],
        streams: dict[str, Stream],
        virtual_computation_graph: ComputationGraph = None,
    ):
        self.device_type = device_type
        self.device_config = device_config_map[device_type]
        assert self.device_config is not None, "Unsupported device type"

        self.global_inputs: dict[int, DTensor] = global_inputs
        self.global_outputs: dict[int, DTensor] = global_outputs
        self.top_function = top_function
        self.core_func_args = core_func_args
        self.streams = streams
        self.virtual_computation_graph: ComputationGraph = virtual_computation_graph

        self.tile_map: dict[str, aie_d.TileOp] = {}
        self.fifo_map: dict[str, aie_d.object_fifo] = {}
        # function name (with id) -> a map from DTensor to fifo name
        self.compute_core_io: dict[str : dict[DTensor, str]] = {}
        self.external_functions: str = ""

        # ------------------------------------------------------------
        # Experimental
        # ------------------------------------------------------------
        self.used_mem_tiles: list[SwitchNode] = []
        self.used_shim_tiles: list[SwitchNode] = []
        self.paths: list[CodeGenerator.DMAPath] = []
        self.function_port_map: dict[str, dict[DTensor, SwitchNode.Port]] = defaultdict(
            lambda: defaultdict(SwitchNode.Port)
        )

        self.fifo_manager: FIFOManager = FIFOManager()
        self.global_dma_trough_port: list[CodeGenerator.GlobalIODMATask] = []
        # ------------------------------------------------------------

        self.aie_module = None  # The top-level AIE IR module
        self.global_ip: aie_ir.InsertionPoint = (
            None  # mark the inserting point for buffers
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
            new_func_inputs = []
            for idx in range(len(func_inputs)):
                if idx in func_args and func_args[idx][0].stream is not None:
                    new_func_inputs.append(func_args[idx][0].stream.allo_element_type)
                    func_inputs[idx] = func_args[idx][0].stream.allo_element_type
                elif idx in func_args and func_args[idx][0].dtensor is not None:
                    new_func_inputs.append(func_inputs[idx])
                else:
                    # fixme: this is a fake placeholder, we'd better remove the useless argument, but doing so leads to crash
                    #           "Cannot destroy a value that still has uses!"
                    new_func_inputs.append(
                        MemRefType.get([], IntegerType.get_signless(8))
                    )

            func_type = allo_func_d.FunctionType.get(
                new_func_inputs,
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
                # fixme: arguments may reuse fifo (ordering and copy-release-acquire)
                # fixme: what about the output??
                compute_core_io = self.compute_core_io
                io_map = (
                    compute_core_io[parsed_function.name.value]
                    if parsed_function.name.value in compute_core_io
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
                alloc_ops = collect_op_by_name(loop, "memref.alloc")
                for alloc_op in alloc_ops:
                    buffer_op = aie_d.BufferOp(
                        buffer=alloc_op.results[0].type,
                        tile=func_core.tile,
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

    def exp_build_core_function(
        self,
        func_core: aie_d.Core,
        original_func: allo_func_d.FuncOp,
        func_args: dict[int, tuple[Argument, bool]],
        arg_to_fifo: dict[int, FIFO],
    ):
        """
        Generate the computation logic for the fake 'while(1)' loop body for an AIE compute core, transforming high-level Allo ops
        into AIE MLIR.

        fixme: current constrain
            - all the argument using the same port has no overlapped liveness range
            - the usage order is aligned to the data transfer order

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
            reused_fifo_name: dict[str, bool] = {}
            for i, argument in enumerate(parsed_function.arguments):
                if not i in func_args:
                    continue
                arg_info: tuple[Argument, bool] = func_args[i]
                if arg_info[0].dtensor is not None:
                    # fixme: argument.uses is unordered??
                    first_use = list(argument.uses)[-1]
                    if first_use is not None:
                        first_use_op = first_use.owner
                        # no branch
                        while first_use_op.parent.name != "func.func":
                            first_use_op = first_use_op.parent
                        fifo = self.fifo_map[arg_to_fifo[i].name]
                        is_input = arg_info[0].dtensor.global_id in self.global_inputs
                        with aie_ir.InsertionPoint(first_use_op):
                            if arg_to_fifo[i].name in reused_fifo_name:
                                fifo.release(1 if is_input else 0, 1)
                            else:
                                reused_fifo_name[arg_to_fifo[i].name] = is_input
                            acquired = fifo.acquire(1 if is_input else 0, 1)
                            # incorrect
                            argument.replace_all_uses_with(acquired)
                else:
                    stream: Stream = arg_info[0].stream
                    fifo = self.fifo_map[stream.name]
                    for use_ in argument.uses:
                        op = use_.owner
                        with aie_ir.InsertionPoint(op.operation):
                            if op.name == "memref.store" or (
                                op.name == "memref.copy" and argument == op.operands[1]
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
                            elif op.name == "memref.copy":  # allo.stream_get, tensor
                                acquired = fifo.acquire(1, 1)
                                op.operands[0] = acquired
                                new_op = op.clone()
                                fifo.release(1, 1)
                                op.erase()
            with aie_ir.InsertionPoint(loop.body):
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
                        tile=func_core.tile,
                        ip=self.global_ip,
                    )
                    for old, new in zip(alloc_op.results, buffer_op.results):
                        old.replace_all_uses_with(new)
                    alloc_op.erase()

                for fifo_name, is_input in reused_fifo_name.items():
                    self.fifo_map[fifo_name].release(1 if is_input else 0, 1)

                aie_scf_d.YieldOp([])
            aie_d.EndOp()

    # ------------------------------------------------------------
    # Data Transfer
    # ------------------------------------------------------------
    class DMAPath:
        def __init__(
            self,
            dtype: str,
            tile_shape: list[int],
            coalesced_size: list[int],
            connected_interfaces: list[list[PEInterface]],
            mem_ports_to_compute: list[SwitchNode.Port],
            tokens: set[str],
        ):
            self.dtype = dtype
            self.tile_shape = tile_shape
            self.coalesced_size = coalesced_size
            self.connected_interfaces = connected_interfaces
            self.mem_ports_to_compute = mem_ports_to_compute
            self.related_tokens = set()
            self.related_tokens.update(tokens)

    @dataclass(frozen=True)
    class GlobalIODMAPort:
        order_tag: int
        fifo: FIFO
        connect_interface: list  # [MulticastInterface]
        size: list[int]
        stride: list[int]
        is_input: bool

    @dataclass(frozen=True)
    class GlobalIODMATask:
        token: str
        start_time: int
        end_time: int
        io_port: "CodeGenerator.GlobalIODMAPort"
        dtensor: DTensor
        size: list[int]
        offset: list[int]

    class DMATaskWithSameToken:
        def __init__(self, task: "CodeGenerator.GlobalIODMATask"):
            self.start_time: int = task.start_time
            self.end_time: int = task.end_time
            self.tasks: list[CodeGenerator.GlobalIODMATask] = [task]

        def add(self, task: "CodeGenerator.GlobalIODMATask"):
            self.start_time = min(self.start_time, task.start_time)
            self.end_time = max(self.end_time, task.end_time)
            self.tasks.append(task)

    def map_data_transfer(self) -> dict[str, dict[int, FIFO]]:
        """
        Construct data transfer path from external memory to each (logical) compute tile.

        TODO: may have influenc on DMA scheduling
        """

        def partition(size: Size4D) -> Size4D:
            """
            Partition the dma task into multiple sub-tasks.
            """
            # find the first none-1 dim
            for dim in range(4):
                if size.get_dim_size(dim) > 1:
                    break
            if dim >= 3:

                raise ValueError("Fail to partition")
            size_part = size.copy()
            partition_size = size.get_dim_size(dim) - 1
            size_part.set_dim_size(dim, partition_size)
            return size_part

        # ------------------------------------------------------------
        MAX_MEM_TILES = self.device_config["mem_tile_num"]
        MAX_SHIM_TILES = self.device_config["shim_tile_num"]
        # ------------------------------------------------------------

        dependencies = self.virtual_computation_graph.get_node_dependencies()
        ordered_nodes: list[str] = []  # topological order
        node_order_tag: dict[str, int] = {}
        tag = 0
        while len(dependencies.items()) > 0:
            tagged_nodes = []
            for node, deps in dependencies.items():
                if len(deps) == 0:
                    node_order_tag[node] = tag
                    tagged_nodes.append(node)
            ordered_nodes.extend(tagged_nodes)
            for node in tagged_nodes:
                del dependencies[node]
                for _, deps in dependencies.items():
                    if node in deps:
                        deps.remove(node)
            tag += 1

        # func name -> (arg idx -> dtensor tiles using that arg)
        global_tensors, arg_idx_to_interface = (
            self.virtual_computation_graph.get_global_io()
        )
        if os.getenv("VERBOSE") == "1":
            print("############## global_tensors ##############")
            for func_name, io_info in global_tensors.items():
                print(func_name)
                for arg_idx, tiles in io_info.items():
                    print("\t", arg_idx)
                    print(tiles)
            print("#############- global_tensors -#############")
        global_dtensor: dict[int, DTensor] = dict(self.global_inputs)
        global_dtensor.update(self.global_outputs)
        global_tile_to_func: dict[int, DTensorTileGroup] = {
            i: DTensorTileGroup("") for i in global_dtensor.keys()
        }
        for func_name, io_info in global_tensors.items():
            for arg_idx, live_dtensor_tiles in io_info.items():
                for tiles in live_dtensor_tiles.dtensor_groups.values():
                    for tile_ in tiles:
                        global_tile_to_func[tile_.tile.dtensor_id].add_tensor_tile(
                            tile_.tile, func_name, arg_idx, live_dtensor_tiles.layout
                        )

        class MulticastInterface:
            """
            MulticastInterface use the same port from source tile
            """

            def __init__(self, interface: PEInterface):
                self.sample_interface: PEInterface = interface
                self.interface_list: set[PEInterface] = {interface}
                sample_global_tensors = global_tensors[interface.pe][
                    interface.interface_idx
                ]
                self.sample_tokens: list[str] = sorted(
                    list(sample_global_tensors.dtensor_groups.keys()),
                    key=string_sort_key,
                )
                # disjoint
                # fixme: better to mentain another set in contiguous ones
                self.tokens: set[tuple[str]] = set()
                self.tokens.add(tuple(self.sample_tokens))

            def _equal_data_transfer(self, other: "MulticastInterface") -> bool:
                if self.sample_interface.layout != other.sample_interface.layout:
                    return False
                sample_global_tensors: LiveDTensorTileGroup = global_tensors[
                    self.sample_interface.pe
                ][self.sample_interface.interface_idx]
                other_global_tensor: LiveDTensorTileGroup = global_tensors[
                    other.sample_interface.pe
                ][other.sample_interface.interface_idx]
                if len(sample_global_tensors.dtensor_groups) == len(
                    other_global_tensor.dtensor_groups
                ):
                    sample_value = next(
                        iter(sample_global_tensors.dtensor_groups.values())
                    )
                    other_value = next(
                        iter(other_global_tensor.dtensor_groups.values())
                    )
                    return sample_value == other_value
                return False

            def _contiguous_data_transfer(
                self, other: "MulticastInterface", current_size: Size4D
            ) -> Size4D:
                for interface in self.interface_list:
                    if interface in other.interface_list:
                        # TODO: can be relaxed
                        return None
                if self.sample_interface.layout != other.sample_interface.layout:
                    return None
                sample_global_tensors: LiveDTensorTileGroup = global_tensors[
                    self.sample_interface.pe
                ][self.sample_interface.interface_idx]
                other_global_tensor: LiveDTensorTileGroup = global_tensors[
                    other.sample_interface.pe
                ][other.sample_interface.interface_idx]
                # TODO: can be relaxed:
                #   currently, if the interface is reused by multiple groups (different tokens), it cannot be multicast
                if len(sample_global_tensors.dtensor_groups) == len(
                    other_global_tensor.dtensor_groups
                ):
                    shape: list[int] = None
                    new_size_list: list[int] = None
                    sample_value_list = [
                        sample_global_tensors.dtensor_groups[k]
                        for k in self.sample_tokens
                    ]
                    new_token_list: list[str] = []
                    for sample_value in sample_value_list:
                        sample_matched_with_other_flag = False
                        for (
                            other_token,
                            other_value,
                        ) in other_global_tensor.dtensor_groups.items():
                            if other_token not in new_token_list and len(
                                sample_value
                            ) == len(other_value):
                                match_flag = True
                                for sample_tile, other_tile in zip(
                                    sample_value, other_value
                                ):
                                    if (
                                        other_tile is None
                                        or sample_tile.tile.dtensor_id
                                        != other_tile.tile.dtensor_id
                                    ):
                                        match_flag = False
                                        break
                                    dtensor = global_dtensor[
                                        sample_tile.tile.dtensor_id
                                    ]
                                    outer_shape = [1, 1, 1, 1]
                                    for i in dtensor.shared_dims:
                                        outer_shape[i] = dtensor.size[i]
                                    if shape is not None and shape != outer_shape:
                                        match_flag = False
                                        break
                                    shape = outer_shape
                                    outer_stride = [1] * 4
                                    for i in reversed(range(3)):
                                        outer_stride[i] = (
                                            outer_stride[i + 1] * outer_shape[i + 1]
                                        )
                                    sample_offset = dtensor.offset_map[
                                        sample_tile.tile.tensor_tile_label
                                    ].to_list()
                                    sample_flattened_idx = sum(
                                        i * s
                                        for i, s in zip(sample_offset, outer_stride)
                                    )
                                    other_offset = dtensor.offset_map[
                                        other_tile.tile.tensor_tile_label
                                    ].to_list()
                                    other_flattened_idx = sum(
                                        i * s
                                        for i, s in zip(other_offset, outer_stride)
                                    )
                                    if other_flattened_idx - sample_flattened_idx != 1:
                                        match_flag = False
                                        break
                                    new_size_list_ = current_size.to_list()
                                    offset_1, offset_2 = sample_offset, other_offset
                                    for i in range(4):
                                        new_size_list_[i] = (
                                            new_size_list_[i]
                                            + offset_2[i]
                                            - offset_1[i]
                                        )
                                    if new_size_list is None:
                                        new_size_list = new_size_list_
                                    elif new_size_list != new_size_list_:
                                        match_flag = False
                                        break
                                if match_flag:
                                    new_token_list.append(other_token)
                                    sample_matched_with_other_flag = True
                                    break
                        if not sample_matched_with_other_flag:
                            return None
                    self.tokens.add(tuple(new_token_list))
                    return Size4D.from_list(new_size_list)
                return None

            def get_pes(self) -> list[str]:
                ret: list[str] = []
                for pe_tile in self.interface_list:
                    ret.append(pe_tile.pe)
                return ret

            def __str__(self):
                return (
                    "["
                    + ", ".join(str(interface) for interface in self.interface_list)
                    + "]"
                )

            def __repr__(self):
                return self.__str__()

        class ContiguousInterface:
            """
            ContiguousInterface always acquire adjacent memory blocks in external memory
            """

            def __init__(self, offset: Offset4D, interface: MulticastInterface):
                self.layout = interface.sample_interface.layout
                self.current_offset: Offset4D = offset
                self.total_size: Size4D = Size4D(1, 1, 1, 1)
                self.interface_list: list[MulticastInterface] = [interface]

            def append(self, offset: Offset4D, other: MulticastInterface) -> bool:
                sample = self.interface_list[-1]
                updated_size = sample._contiguous_data_transfer(other, self.total_size)
                if updated_size is None:
                    return False
                self.interface_list.append(other)
                self.current_offset = offset
                self.total_size = updated_size
                return True

            def __str__(self):
                return "; ".join(str(interface) for interface in self.interface_list)

            def __repr__(self):
                return self.__str__()

        def assign_mem_tile(
            dtype: str,
            interface_list: list[MulticastInterface],
            is_input: bool,
            coalesced_size: Size4D,
            tile_size: Size4D,
            tile_shape: list[int],
        ):
            """
            Assign a memory tile to the given dtensor tiles.
            If no memory tile is available, return None.
            Else, return the assigned memory tile, the port id to shim, and the port ids to compute.
            """
            send_need = len(interface_list) if is_input else 1
            recv_need = (
                1
                if is_input
                else sum(len(group.interface_list) for group in interface_list)
            )
            send_shape: list[int] = tile_shape if is_input else coalesced_size.to_list()
            recv_shape: list[int] = coalesced_size.to_list() if is_input else tile_shape
            tile_total_size = tile_size.get_total_size()
            connected_interfaces: list[list[PEInterface]] = []
            for multicast_interface in interface_list:
                if is_input:
                    connected_interfaces.append(
                        list(multicast_interface.interface_list)
                    )
                else:
                    for pe_interface in multicast_interface.interface_list:
                        connected_interfaces.append([pe_interface])
            if os.getenv("VERBOSE") == "1":
                print(f"send_need: {send_need}, recv_need: {recv_need}")
            assigned_mem_tile = None
            # Attempt to use a new memory tile
            if (
                len(self.used_mem_tiles) < MAX_MEM_TILES
                and send_need <= Config.MEM_MAX_SEND
                and recv_need <= Config.MEM_MAX_RECV
            ):
                assigned_mem_tile = SwitchNode(
                    name=f"{len(self.used_mem_tiles)}_mem_tile",
                    send_port_num=Config.MEM_MAX_SEND,
                    recv_port_num=Config.MEM_MAX_RECV,
                )
                self.used_mem_tiles.append(assigned_mem_tile)
            else:
                # Attempt to use an existing memory tile
                for mem_tile in self.used_mem_tiles:
                    if (
                        len(mem_tile.send_ports) + send_need <= Config.MEM_MAX_SEND
                        and len(mem_tile.recv_ports) + recv_need <= Config.MEM_MAX_RECV
                    ):
                        assigned_mem_tile = mem_tile
                        break
            # Use new ports
            if assigned_mem_tile is not None:
                send_ports, recv_ports = [], []
                for i in range(send_need):
                    port = SwitchNode.Port(
                        port_id=len(assigned_mem_tile.send_ports) + i,
                        data_shape=send_shape,
                        dtype=dtype,
                        connected_nodes=interface_list[i].get_pes() if is_input else [],
                    )
                    send_ports.append(port)
                if is_input:
                    for i in range(recv_need):
                        port = SwitchNode.Port(
                            port_id=len(assigned_mem_tile.recv_ports) + i,
                            data_shape=recv_shape,
                            dtype=dtype,
                            connected_nodes=[],
                        )
                        recv_ports.append(port)
                else:
                    for multicast_interface in interface_list:
                        for pe_interface in multicast_interface.interface_list:
                            port = SwitchNode.Port(
                                port_id=len(assigned_mem_tile.recv_ports)
                                + len(recv_ports),
                                data_shape=recv_shape,
                                dtype=dtype,
                                connected_nodes=[pe_interface.pe],
                            )
                            recv_ports.append(port)
                assigned_mem_tile.intra_connect.append(
                    SwitchNode.IntraConnect(
                        [port.id for port in send_ports],
                        [port.id for port in recv_ports],
                        list(
                            range(
                                0,
                                max(send_need, recv_need) * tile_total_size,
                                tile_total_size,
                            )
                        ),
                    )
                )
                if os.getenv("VERBOSE") == "1":
                    print("\nassigned_mem_tile: ", end="")
                    assigned_mem_tile.print()
                return (
                    assigned_mem_tile,
                    send_ports,
                    recv_ports,
                    connected_interfaces,
                )
            return None, -1, [], connected_interfaces

        def assign_shim_tile(
            mem_tile: SwitchNode,
            mem_port: SwitchNode.Port,
            is_input: bool,
        ):
            """
            Assign a shim tile connected to a mem tile port.
            If no shim tile is available, return None.
            Else, return the assigned shim tile, and the shim port id.
            """
            send_need = 1 if is_input else 0
            recv_need = 0 if is_input else 1
            connected_mem = [mem_tile.name]
            assigned_shim_tile = None
            if len(mem_port.connected_nodes) > 0:
                assert len(mem_port.connected_nodes) == 1
                for shim_tile in self.used_shim_tiles:
                    if shim_tile.name == mem_port.connected_nodes[0]:
                        if is_input:
                            for idx, port in enumerate(shim_tile.send_ports):
                                if port.connected_nodes == connected_mem:
                                    return shim_tile, idx
                        else:
                            for idx, port in enumerate(shim_tile.recv_ports):
                                if port.connected_nodes == connected_mem:
                                    return shim_tile, idx
                raise ValueError("Run into an unreachable point")
            # Attempt to use a new shim tile
            if len(self.used_shim_tiles) < MAX_SHIM_TILES:
                assigned_shim_tile = SwitchNode(
                    name=f"{len(self.used_shim_tiles)}_shim_tile",
                    send_port_num=Config.SHIM_MAX_SEND,
                    recv_port_num=Config.SHIM_MAX_RECV,
                )
                self.used_shim_tiles.append(assigned_shim_tile)
            else:
                for shim_tile in self.used_shim_tiles:
                    if (
                        len(shim_tile.send_ports) + send_need <= Config.SHIM_MAX_SEND
                        and len(shim_tile.recv_ports) + recv_need
                        <= Config.SHIM_MAX_RECV
                    ):
                        assigned_shim_tile = shim_tile
                        break
            # Use new ports
            if assigned_shim_tile is not None:
                if is_input:
                    send_port = SwitchNode.Port(
                        port_id=len(assigned_shim_tile.send_ports),
                        data_shape=mem_port.data_shape,
                        dtype=mem_port.dtype,
                        connected_nodes=connected_mem,
                    )
                    assigned_shim_tile.send_ports.append(send_port)
                else:
                    recv_port = SwitchNode.Port(
                        port_id=len(assigned_shim_tile.recv_ports),
                        data_shape=mem_port.data_shape,
                        dtype=mem_port.dtype,
                        connected_nodes=connected_mem,
                    )
                    assigned_shim_tile.recv_ports.append(recv_port)
                mem_port.connected_nodes.append(assigned_shim_tile.name)
                if os.getenv("VERBOSE") == "1":
                    print("\nassigned_shim_tile: ", end="")
                    assigned_shim_tile.print()
                return assigned_shim_tile, send_port.id if is_input else recv_port.id
            return None, -1

        def assign_tiles(
            contiguous_interface: list[MulticastInterface],
            total_size: Size4D,
            tile_size: Size4D,
            is_input: bool,
            tile_param_type: list,
        ) -> bool:
            """
            Assign a shim tile and mem tile for contiguous interfaces (send/receive data contiguous in external memory).
            Return True if the assignment succeeded, otherwise return False.
            """
            coalesced_size = Size4D.coalesce(total_size, tile_size)
            tokens: set[str] = set()
            for multicast_interface in contiguous_interface:
                for token_tuple in multicast_interface.tokens:
                    for token in token_tuple:
                        tokens.add(token)
            (
                assigned_mem_tile,
                send_ports,
                recv_ports,
                connected_interfaces,
            ) = assign_mem_tile(
                tile_dtype,
                contiguous_interface,
                is_input,
                coalesced_size,
                tile_size,
                tile_param_type,
            )
            if assigned_mem_tile is not None:
                mem_port_to_shim: SwitchNode.Port = (
                    recv_ports[0] if is_input else send_ports[0]
                )
                ports_to_compute: list[SwitchNode.Port] = (
                    send_ports if is_input else recv_ports
                )
                assigned_shim_tile, shim_port_id = assign_shim_tile(
                    assigned_mem_tile,
                    mem_port_to_shim,
                    is_input,
                )
                if assigned_shim_tile is None:
                    # invalidate the intra_connect
                    assigned_mem_tile.intra_connect.pop()
                    # raise ValueError("Fail to assign shim tile")
                else:
                    assigned_mem_tile.send_ports.extend(send_ports)
                    assigned_mem_tile.recv_ports.extend(recv_ports)
                    path = CodeGenerator.DMAPath(
                        tile_dtype,
                        tile_param_type,
                        coalesced_size.to_list(),
                        connected_interfaces,
                        ports_to_compute,
                        tokens,
                    )
                    self.paths.append(path)
                    assert len(connected_interfaces) == len(ports_to_compute)
                    for idx, mem_port_to_compute in enumerate(ports_to_compute):
                        if is_input:
                            dma_fifo = self.fifo_manager.create_fifo(
                                src=assigned_mem_tile.name,
                                dst=mem_port_to_compute.connected_nodes,
                                data_shape=mem_port_to_compute.data_shape,
                                dtype=mem_port_to_compute.dtype,
                                dimensions_to_stream=transfer_layout,
                            )
                        else:
                            assert len(mem_port_to_compute.connected_nodes) == 1
                            dma_fifo = self.fifo_manager.create_fifo(
                                src=mem_port_to_compute.connected_nodes[0],
                                dst=[assigned_mem_tile.name],
                                data_shape=mem_port_to_compute.data_shape,
                                dtype=mem_port_to_compute.dtype,
                            )
                        mem_port_to_compute.bind_to_fifo(dma_fifo)
                        for interface in connected_interfaces[idx]:
                            mapped_interface[interface.pe][
                                interface.interface_idx
                            ] = mem_port_to_compute.bind_fifo
                    shim_port_to_mem = (
                        assigned_shim_tile.send_ports[shim_port_id]
                        if is_input
                        else assigned_shim_tile.recv_ports[shim_port_id]
                    )
                    if is_input:
                        dma_fifo = self.fifo_manager.create_fifo(
                            src=assigned_shim_tile.name,
                            dst=shim_port_to_mem.connected_nodes,
                            data_shape=shim_port_to_mem.data_shape,
                            dtype=shim_port_to_mem.dtype,
                        )
                    else:
                        assert (
                            len(shim_port_to_mem.connected_nodes) == 1
                            and shim_port_to_mem.connected_nodes[0]
                            == assigned_mem_tile.name
                        )
                        dma_fifo = self.fifo_manager.create_fifo(
                            src=assigned_mem_tile.name,
                            dst=[assigned_shim_tile.name],
                            data_shape=shim_port_to_mem.data_shape,
                            dtype=shim_port_to_mem.dtype,
                            dimensions_to_stream=transfer_layout,
                        )
                    shim_port_to_mem.bind_to_fifo(dma_fifo)
                    mem_port_to_shim.bind_to_fifo(dma_fifo)
                    order_tag = len(node_order_tag)
                    for multicase_interfaces in contiguous_interface:
                        for interface in multicase_interfaces.interface_list:
                            if order_tag > node_order_tag[interface.pe]:
                                order_tag = node_order_tag[interface.pe]
                    global_io_port.append(
                        CodeGenerator.GlobalIODMAPort(
                            order_tag=order_tag,
                            fifo=dma_fifo,
                            connect_interface=contiguous_interface,
                            size=coalesced_size.to_list(),
                            stride=dtensor.stride,
                            is_input=is_input,
                        )
                    )
                    return True
            # reuse path
            for path in self.paths:
                if (
                    path.dtype == tile_dtype
                    and path.tile_shape == tile_param_type
                    and path.coalesced_size == coalesced_size.to_list()
                    and path.connected_interfaces == connected_interfaces
                ):
                    if path.related_tokens.isdisjoint(tokens):
                        path.related_tokens.update(tokens)
                        for idx, mem_port_to_compute in enumerate(
                            path.mem_ports_to_compute
                        ):
                            for interface in connected_interfaces[idx]:
                                mapped_interface[interface.pe][
                                    interface.interface_idx
                                ] = mem_port_to_compute.bind_fifo
                        return True
            return False

        mapped_interface: dict[str, dict[int, FIFO]] = {
            i: {} for i in global_tensors.keys()
        }
        global_io_port: list[CodeGenerator.GlobalIODMAPort] = []
        for idx, dtensor_tile_group in global_tile_to_func.items():
            dtensor = global_dtensor[idx]
            # transfer tile meta data
            is_input = idx in self.global_inputs
            tile_dtype = dtensor.dtype
            tile_param_type = dtensor.type_as_param
            tile_shape = list(dtensor.size)
            for i in dtensor.shared_dims:
                tile_shape[i] = 1
            tile_size = Size4D.from_list(tile_shape)
            # key: offset specific to dtensor
            unresolved_tile: dict[Offset4D, list[MulticastInterface]] = {}
            in_process: set[PEInterface] = set()
            for (
                dtensor_tile,
                interface_list,
            ) in dtensor_tile_group.dtensor_tile_to_pe_interfaces.items():
                unresolved: set[PEInterface] = set()
                for interface in interface_list:
                    if (
                        interface.interface_idx not in mapped_interface[interface.pe]
                        and interface not in in_process
                    ):
                        unresolved.add(interface)
                        in_process.add(interface)
                multicast_list: list[MulticastInterface] = [
                    MulticastInterface(interface) for interface in unresolved
                ]
                changed = True
                while changed:
                    changed = False
                    new_list = []
                    used = [False] * len(multicast_list)
                    for i in range(len(multicast_list)):
                        if used[i]:
                            continue
                        current = multicast_list[i]
                        for j in range(i + 1, len(multicast_list)):
                            if used[j]:
                                continue
                            if current._equal_data_transfer(multicast_list[j]):
                                current.interface_list.update(
                                    multicast_list[j].interface_list
                                )
                                current.tokens.update(multicast_list[j].tokens)
                                used[j] = True
                                changed = True
                        new_list.append(current)
                    multicast_list = new_list
                if len(multicast_list) > 0:
                    unresolved_tile[
                        dtensor.offset_map[dtensor_tile.tensor_tile_label]
                    ] = multicast_list
            # coalesced access pattern on dtensor will give a hint
            coalesced_access_pattern, coalesce_info, coalesced_multicast_interfaces = (
                coalesce_memory_access(unresolved_tile)
            )
            if os.getenv("VERBOSE") == "1":
                print("<<<<< coalesced_multicast_interfaces >>>>>")
                print(coalesced_multicast_interfaces)
                print("===== coalesced_multicast_interfaces =====")
            contiguous_interfaces: list[ContiguousInterface] = []
            for start_offset in coalesced_access_pattern.keys():
                coalesced_interfaces: list[list[MulticastInterface]] = (
                    coalesced_multicast_interfaces[start_offset]
                )
                left = 0
                while left < len(coalesced_interfaces):
                    next_flag = True
                    for left_i, multicast_interface in enumerate(
                        coalesced_interfaces[left]
                    ):
                        if multicast_interface is not None:
                            next_flag = False
                            contiguous: ContiguousInterface = ContiguousInterface(
                                coalesce_info[start_offset][left], multicast_interface
                            )
                            coalesced_interfaces[left][left_i] = None
                            right = left + 1
                            continue_flag = True
                            while continue_flag and right < len(coalesced_interfaces):
                                continue_flag = False
                                for right_i, next_interface in enumerate(
                                    coalesced_interfaces[right]
                                ):
                                    if next_interface is not None:
                                        if contiguous.append(
                                            coalesce_info[start_offset][right],
                                            next_interface,
                                        ):
                                            continue_flag = True
                                            coalesced_interfaces[right][right_i] = None
                                            right = right + 1
                                            break
                            contiguous_interfaces.append(contiguous)
                    if next_flag:
                        left += 1
            if os.getenv("VERBOSE") == "1":
                print("\n<<<<< contiguous_interfaces >>>>>")
                for contiguous_interface in contiguous_interfaces:
                    print(contiguous_interface)
                print("===== contiguous_interfaces =====\n")
            for contiguous_interface in contiguous_interfaces:
                interface_list: list[MulticastInterface] = (
                    contiguous_interface.interface_list
                )
                size = contiguous_interface.total_size
                transfer_layout = contiguous_interface.layout
                while size.get_total_size() != 0:
                    if assign_tiles(
                        interface_list, size, tile_size, is_input, tile_param_type
                    ):
                        break
                    size_cp = size.copy()
                    # keep partitioning until success
                    while True:
                        partitioned_size = partition(size_cp)
                        partitioned_interface_list = interface_list[
                            : partitioned_size.get_total_size()
                        ]
                        if assign_tiles(
                            partitioned_interface_list,
                            partitioned_size,
                            tile_size,
                            is_input,
                            tile_param_type,
                        ):
                            break
                        size_cp = partitioned_size
                    size = Size4D.subtract(size, partitioned_size)
                    inc = partitioned_size.get_total_size()
                    interface_list = interface_list[inc:]

        token_map: dict[str, str] = {}
        token_cnt = 0
        related_token_list: list[set[tuple[str]]] = []
        for io_port in global_io_port:
            interfaces: list[MulticastInterface] = io_port.connect_interface
            related_tokens: set[tuple[str]] = set()
            for interface in interfaces:
                related_tokens.update(interface.tokens)
            related_token_list.append(related_tokens)
        merged_token_sets: list[set[tuple[str]]] = merge_token_sets(related_token_list)
        for token_set in merged_token_sets:
            token_number = len(list(token_set)[0])
            for token_idx in range(token_number):
                token = f"token_{token_cnt + token_idx}"
                for local_tokens in token_set:
                    assert len(local_tokens) == token_number
                    token_map[local_tokens[token_idx]] = token
            token_cnt += token_number

        for io_port in global_io_port:
            interfaces: list[MulticastInterface] = io_port.connect_interface
            # TODO: only support limited cases (need to use assert as guard)
            # TODO: related to DMA scheduling
            tensor_tile_group = global_tensors[interfaces[0].sample_interface.pe][
                interfaces[0].sample_interface.interface_idx
            ]
            for live_tensor_tiles in tensor_tile_group.dtensor_groups.values():
                for live_tensor_tile in live_tensor_tiles:
                    dtensor_ = global_dtensor[live_tensor_tile.tile.dtensor_id]
                    self.global_dma_trough_port.append(
                        CodeGenerator.GlobalIODMATask(
                            token=token_map[live_tensor_tile.token],
                            start_time=live_tensor_tile.first_use
                            + Config.GLOBAL_CODE_OFFSET * io_port.order_tag,
                            end_time=live_tensor_tile.last_use
                            + Config.GLOBAL_CODE_OFFSET * io_port.order_tag,
                            io_port=io_port,
                            dtensor=dtensor_,
                            size=io_port.size,
                            offset=dtensor_.offset_map[
                                live_tensor_tile.tile.tensor_tile_label
                            ].to_list(),
                        )
                    )
        if os.getenv("VERBOSE") == "1":
            print("## global_dma_trough_port")
            for ele in self.global_dma_trough_port:
                print(ele)
            print()
        global_arg_idx_to_interface: dict[str, dict[int, FIFO]] = {
            i: {} for i in global_tensors.keys()
        }
        for func_name, interface_map in mapped_interface.items():
            dict_: dict[int, FIFO] = {}
            for idx, interface in arg_idx_to_interface[func_name].items():
                dict_[idx] = interface_map[interface]
            global_arg_idx_to_interface[func_name] = dict_
        return global_arg_idx_to_interface

    # ------------------------------------------------------------
    # Compute Tile
    # ------------------------------------------------------------

    def map_core_func_to_physical_tiles(self) -> dict[str, tuple[int, int]]:
        """
        Map the core functions to physical tiles.
        TODO:
            - mapping strategies should be selected by cost
            - careful with nodes with multiple inputs/outputs
              (if ports are exceeded, we should try to assign them to adjacent compute tiles to share local memory)
        """
        core_fucn_mapping: dict[str, tuple[int, int]] = {}
        mesh_shape = self.device_config["mesh"]
        max_row, max_col = mesh_shape
        tile_used = np.zeros(mesh_shape, dtype=bool)
        # connected nodes are grouped into chains when COMPUTE_TILE_WITH_SHARED_MEMORY == 2
        if Config.COMPUTE_TILE_WITH_SHARED_MEMORY == 2:

            class NodeDeque:
                def __init__(self, node_name: str):
                    self.nodes: list[str] = [node_name]

                @staticmethod
                def connect(
                    node_1: str,
                    node_2: str,
                    node_deque1: "NodeDeque",
                    node_deque2: "NodeDeque",
                ):
                    if node_2 == node_deque2.nodes[-1]:
                        node_1, node_2 = node_2, node_1
                        node_deque1, node_deque2 = node_deque2, node_deque1
                    if node_1 == node_deque1.nodes[0]:
                        node_deque1.nodes.reverse()
                    if node_2 == node_deque2.nodes[-1]:
                        node_deque2.nodes.reverse()
                    node_deque1.nodes.extend(node_deque2.nodes)
                    return node_deque1

                def __str__(self):
                    return f"NodeDeque({self.nodes})"

                def __repr__(self):
                    return self.__str__()

            connection_info = self.virtual_computation_graph.get_connections()
            connection_info.sort(key=lambda x: x[0], reverse=True)
            names = self.virtual_computation_graph.nodes.keys()
            grouped_nodes: dict[str, NodeDeque] = {
                name: NodeDeque(name) for name in names
            }
            for connection in connection_info:
                grouped_a, grouped_b = (
                    grouped_nodes[connection[1]],
                    grouped_nodes[connection[2]],
                )
                if grouped_a is None or grouped_b is None:
                    continue
                new_group = NodeDeque.connect(
                    connection[1], connection[2], grouped_a, grouped_b
                )
                grouped_nodes.pop(connection[1])
                grouped_nodes.pop(connection[2])
                grouped_nodes[new_group.nodes[0]] = new_group
                grouped_nodes[new_group.nodes[-1]] = new_group
            # TODO: map nodes according to global io
            # heuristic
            traverse_idx = 0
            sorted_values = [
                grouped_nodes[key]
                for key in sorted(grouped_nodes.keys(), key=string_sort_key)
            ]
            assigned = set()
            for deque in sorted_values:
                if deque in assigned:
                    continue
                assigned.add(deque)
                head = deque.nodes[0]
                while tile_used[traverse_idx // max_col][traverse_idx % max_col]:
                    traverse_idx += 1
                    if traverse_idx >= max_row * max_col:
                        raise ValueError("Too many nodes")
                col_idx = traverse_idx % max_col
                row_idx = traverse_idx // max_col
                core_fucn_mapping[head] = (row_idx, col_idx)
                tile_used[row_idx][col_idx] = True
                reverse = False
                for node in deque.nodes[1:]:
                    while tile_used[row_idx][col_idx]:
                        if reverse:
                            row_idx -= 1
                            if row_idx < 0:
                                row_idx = 0
                                col_idx += 1
                                reverse = not reverse
                        else:
                            row_idx += 1
                            if row_idx >= max_row:
                                row_idx = max_row - 1
                                col_idx += 1
                                reverse = not reverse
                    core_fucn_mapping[node] = (row_idx, col_idx)
                    tile_used[row_idx][col_idx] = True
            if os.getenv("VERBOSE") == "1":
                print("<<< Mapping >>>")
                for node, (row, col) in core_fucn_mapping.items():
                    print(f"{node}: ({row}, {col})")
                print()
            return core_fucn_mapping
        raise ValueError("To be implemented")

    # ############################################################
    # AIE Code Generation
    # ############################################################
    def aie_codegen_experimental(
        self,
        core_funcs: list[allo_func_d.FuncOp],
        external_funcs: list[allo_func_d.FuncOp],
    ) -> aie_ir.Module:
        # mapping to physical/logical
        # TODO: co-designed mapping to different types of tiles
        arg_to_fifo = self.map_data_transfer()
        core_function_mapping = self.map_core_func_to_physical_tiles()
        for func in external_funcs:
            self.external_functions += format_str(str(func), indent=4)

        wrapper_code = f"""
            module {{
                aie.device({self.device_type}) {{
        """
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
                # shim tiles
                for i, shim_tile in enumerate(self.used_shim_tiles):
                    self.tile_map[shim_tile.name] = aie_d.TileOp(col=i, row=0)
                # mem tiles
                for i, mem_tile in enumerate(self.used_mem_tiles):
                    self.tile_map[mem_tile.name] = aie_d.TileOp(col=i, row=1)
                # compute tiles
                for func_name, (row, col) in core_function_mapping.items():
                    self.tile_map[func_name] = aie_d.TileOp(col=col, row=row + 2)
                # define fifos
                # - stream fifos: compute <-> compute
                for stream_name, stream in self.streams.items():
                    self.fifo_map[stream_name] = aie_d.object_fifo(
                        stream_name,
                        self.tile_map[stream.src],
                        self.tile_map[stream.dst],
                        depth=stream.type.depth,
                        datatype=aie_ir.MemRefType.get(
                            stream.type.shape,
                            get_element_type(str(stream.type.dtype)),
                        ),
                    )
                # - io fifos: shim <-> mem <-> compute
                for dma_fifo in self.fifo_manager.fifos:
                    self.fifo_map[dma_fifo.name] = aie_d.object_fifo(
                        dma_fifo.name,
                        self.tile_map[dma_fifo.src],
                        [self.tile_map[node] for node in dma_fifo.dst],
                        depth=dma_fifo.depth,
                        datatype=aie_ir.MemRefType.get(
                            dma_fifo.data_shape,
                            get_element_type(str(dma_fifo.dtype)),
                        ),
                        dimensionsToStream=dma_fifo.dimensions_to_stream,
                    )
                # link fifos: in aie, mem tile serves as the linkages
                for dma_node in self.used_mem_tiles:
                    for connect in dma_node.intra_connect:
                        producer = [
                            self.fifo_map[
                                dma_node.recv_ports[recv_port_id].bind_fifo.name
                            ]
                            for recv_port_id in connect.recv_port_ids
                        ]
                        consumer = [
                            self.fifo_map[
                                dma_node.send_ports[send_port_id].bind_fifo.name
                            ]
                            for send_port_id in connect.send_port_ids
                        ]
                        # fixme: is it possible that both producer and consumer are not single fifo?
                        producer_offset = [] if len(producer) == 1 else connect.offsets
                        consumer_offset = [] if len(consumer) == 1 else connect.offsets
                        aie_d.object_fifo_link(
                            producer, consumer, producer_offset, consumer_offset
                        )
                # compute logic on each compute tile
                for func in core_funcs:
                    func_name = func.attributes["sym_name"].value
                    use_external_kernel = self.virtual_computation_graph.nodes[
                        func_name
                    ].meta_data.use_external_kernel
                    func_core = aie_d.Core(
                        tile=self.tile_map[func_name],
                        link_with=("external.o" if use_external_kernel else None),
                    )
                    if self.global_ip is None:
                        self.global_ip = aie_ir.InsertionPoint(func_core)
                    self.exp_build_core_function(
                        func_core,
                        func,
                        self.core_func_args[func_name],
                        arg_to_fifo[func_name],
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
                # TODO: need more robust and smart DMA scheduling
                dma_task_groups: dict[str, CodeGenerator.DMATaskWithSameToken] = {}
                for global_dma in self.global_dma_trough_port:
                    if global_dma.token not in dma_task_groups:
                        group = CodeGenerator.DMATaskWithSameToken(global_dma)
                        dma_task_groups[global_dma.token] = group
                    else:
                        group = dma_task_groups[global_dma.token]
                        group.add(global_dma)
                runtime_seq_entry_block = runtime_seq.body.blocks.append(*runtime_args)
                with aie_ir.InsertionPoint(runtime_seq_entry_block):
                    # data with same token should be transferred together
                    # fixme: if execution fails with runtime_error, possibly because the transfer order leads to 'deadlock'
                    task_groups: list[CodeGenerator.DMATaskWithSameToken] = list(
                        dma_task_groups.values()
                    )
                    task_groups.sort(key=lambda x: x.start_time)
                    dma_bd_map: dict[str, int] = {
                        shim_tile.name: 0 for shim_tile in self.used_shim_tiles
                    }
                    launched_dma_to_external: list[aie_d.object_fifo] = []
                    # launch a group of tasks with the same token
                    for task_group in task_groups:
                        task_group.tasks.sort(key=lambda x: x.start_time)
                        fifo_to_tasks: dict[
                            str, list[CodeGenerator.GlobalIODMATask]
                        ] = defaultdict(list)
                        for global_dma in task_group.tasks:
                            fifo_to_tasks[global_dma.io_port.fifo.name].append(
                                global_dma
                            )
                        updated = True
                        while updated:
                            updated = False
                            coalesced_fifo_to_tasks: dict[
                                str, list[CodeGenerator.GlobalIODMATask]
                            ] = defaultdict(list)
                            for fifo, tasks in fifo_to_tasks.items():
                                left = 0
                                while left < len(tasks):
                                    base_size = [1, 1, 1, 1]
                                    inc_idx = None
                                    current_offset = Offset4D(
                                        tasks[left].offset[0],
                                        tasks[left].offset[1],
                                        tasks[left].offset[2],
                                        tasks[left].offset[3],
                                    )
                                    right = left + 1
                                    while right < len(tasks):
                                        if tasks[left].dtensor == tasks[right].dtensor:
                                            incomming_offset = Offset4D(
                                                tasks[right].offset[0],
                                                tasks[right].offset[1],
                                                tasks[right].offset[2],
                                                tasks[right].offset[3],
                                            )
                                            idx = current_offset.check_next_offset(
                                                incomming_offset
                                            )
                                            if idx >= 0 and (
                                                inc_idx is None or inc_idx == idx
                                            ):
                                                # dma size constrain
                                                max_size = (
                                                    Config.SHIM_DMA_HARDWARE_MAX_SIZES[
                                                        idx
                                                    ]
                                                )
                                                if (
                                                    max_size > 0
                                                    and max_size
                                                    < (base_size[idx] + 1)
                                                    * tasks[left].size[idx]
                                                ):
                                                    break
                                                inc_idx = idx
                                                base_size[idx] += 1
                                                current_offset = incomming_offset
                                                right += 1
                                            else:
                                                break
                                    for i in range(4):
                                        if base_size[i] > 1:
                                            updated = True
                                        base_size[i] *= tasks[left].size[i]
                                    coalesced_task = CodeGenerator.GlobalIODMATask(
                                        tasks[left].token,
                                        tasks[left].start_time,
                                        tasks[right - 1].end_time,
                                        tasks[left].io_port,
                                        tasks[left].dtensor,
                                        base_size,
                                        tasks[left].offset,
                                    )
                                    coalesced_fifo_to_tasks[fifo].append(coalesced_task)
                                    left = right
                            fifo_to_tasks = coalesced_fifo_to_tasks
                        coalesced_tasks: list[CodeGenerator.GlobalIODMATask] = []
                        for tasks in fifo_to_tasks.values():
                            coalesced_tasks.extend(tasks)
                        coalesced_tasks.sort(key=lambda x: x.start_time)
                        dma_bd_workload: dict[str, int] = {
                            shim_tile.name: 0 for shim_tile in self.used_shim_tiles
                        }
                        # check buffer descriptor workload
                        overload_flag = False
                        for global_dma in coalesced_tasks:
                            used_shim = (
                                global_dma.io_port.fifo.src
                                if global_dma.io_port.is_input
                                else global_dma.io_port.fifo.dst[0]
                            )
                            dma_bd_workload[used_shim] += 1
                            if (
                                dma_bd_workload[used_shim] + dma_bd_map[used_shim]
                                >= Config.DMA_MAX_BDS
                            ):
                                overload_flag = True
                                break
                        # sync on output before reusing buffer descriptor (https://github.com/Xilinx/mlir-aie/blob/main/programming_guide/section-2/section-2d/DMATasks.md#best-practices-for-data-movement-and-synchronization-with-npu_dma_memcpy_nd)
                        # TODO: better analysis to decide the 'sync point'
                        if overload_flag:
                            for fifo in launched_dma_to_external:
                                aiex_d.dma_wait(fifo)
                            launched_dma_to_external.clear()
                            for shim_name in dma_bd_map.keys():
                                dma_bd_map[shim_name] = 0
                        for global_dma in coalesced_tasks:
                            dma_fifo = self.fifo_map[global_dma.io_port.fifo.name]
                            used_shim = (
                                global_dma.io_port.fifo.src
                                if global_dma.io_port.is_input
                                else global_dma.io_port.fifo.dst[0]
                            )
                            bd_id = dma_bd_map[used_shim]
                            assert (
                                bd_id < Config.DMA_MAX_BDS
                            ), "each shim tile have at most 16 buffer descriptor"
                            dma_bd_map[used_shim] += 1
                            aiex_d.NpuDmaMemcpyNd(
                                metadata=dma_fifo,
                                bd_id=bd_id,
                                mem=runtime_seq_entry_block.arguments[
                                    global_dma.dtensor.global_id
                                ],
                                offsets=global_dma.offset,
                                sizes=global_dma.size,
                                strides=global_dma.io_port.stride,
                                issue_token=True,
                            )
                            if not global_dma.io_port.is_input:
                                launched_dma_to_external.append(dma_fifo)
                    for launched_fifo in launched_dma_to_external:
                        aiex_d.dma_wait(launched_fifo)

                    aie_d.EndOp()

        return self.aie_module

    def aie_codegen(
        self,
        core_func_groups: dict[str, list[allo_func_d.FuncOp]],
        external_funcs: list[allo_func_d.FuncOp],
        inputs,
        outputs,
        use_external_kernels: bool,
        trace: list[tuple[str, tuple[int, ...]]],
        trace_size: int,
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

            available_shim_for_trace: set[str] = set()
            with aie_ir.InsertionPoint(end_op):
                # shim tile
                for shim_id in range(shim_tile_num):
                    self.tile_map[f"shim_{shim_id}"] = aie_d.TileOp(col=shim_id, row=0)
                    available_shim_for_trace.add(f"shim_{shim_id}")
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
                aie_mesh = self.device_config["mesh"]
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
                                if io == "out":
                                    available_shim_for_trace.remove(
                                        f"shim_{dma_tile.shim_id}"
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
                                    name_suffix = "".join(
                                        f"_{ele}" for ele in tensor_tile
                                    )
                                    name = f"{io}_mem_{dtensor.name}_{name_suffix}"
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
                for stream_name, stream in self.streams.items():
                    src_tile = self.tile_map[f"compute_{stream.src}"]
                    dst_tile = [self.tile_map[f"compute_{stream.dst}"]]
                    self.fifo_map[stream_name] = aie_d.object_fifo(
                        stream_name,
                        src_tile,
                        dst_tile,
                        depth=stream.type.depth,
                        datatype=aie_ir.MemRefType.get(
                            stream.type.shape,
                            get_element_type(str(stream.type.dtype)),
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
                            func_core, func, self.core_func_args[func_name_w_id]
                        )

                @dataclass
                class TraceInfo:
                    traced_tile_idx: tuple[int]
                    shim_tile_idx: tuple[int]
                    packet_id: int

                total_transfer = 0
                for tiles in io_mapping.values():
                    total_transfer += len(tiles)
                assert (
                    total_transfer <= Config.DMA_MAX_BDS
                ), "Exceed total buffer descriptor number."
                enabled_trace: list[TraceInfo] = []
                # fixme: can be relaxed
                if (
                    trace is not None
                    and total_transfer < Config.DMA_MAX_BDS
                    and len(available_shim_for_trace) > 0
                ):
                    trace_transfer_shim_tile = self.tile_map[
                        next(iter(available_shim_for_trace))
                    ]
                    packet_id = 0
                    for traced_tile in trace:
                        packet_id += 1
                        if packet_id + total_transfer > Config.DMA_MAX_BDS:
                            break
                        func_name = (
                            traced_tile[0]
                            + f"_{'_'.join([str(x) for x in traced_tile[1]])}"
                        )
                        compute_tile = self.tile_map[f"compute_{func_name}"]
                        aie_d.packetflow(
                            packet_id,
                            compute_tile,
                            9,  # WireBundle: Trace = 9
                            0,
                            trace_transfer_shim_tile,
                            1,  # WireBundle: DMA = 1
                            1,
                            True,
                        )
                        enabled_trace.append(
                            TraceInfo(
                                (compute_tile.col.value, compute_tile.row.value),
                                (
                                    trace_transfer_shim_tile.col.value,
                                    trace_transfer_shim_tile.row.value,
                                ),
                                packet_id,
                            )
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
                    for trace_info in enabled_trace:
                        aiex_d.npu_write32(
                            213200,
                            2038038528,
                            column=trace_info.traced_tile_idx[0],
                            row=trace_info.traced_tile_idx[1],
                        )
                        aiex_d.npu_write32(
                            213204,
                            trace_info.packet_id,
                            column=trace_info.traced_tile_idx[0],
                            row=trace_info.traced_tile_idx[1],
                        )
                        aiex_d.npu_write32(
                            213216,
                            1260724769,
                            column=trace_info.traced_tile_idx[0],
                            row=trace_info.traced_tile_idx[1],
                        )
                        aiex_d.npu_write32(
                            213220,
                            439168079,
                            column=trace_info.traced_tile_idx[0],
                            row=trace_info.traced_tile_idx[1],
                        )
                        aiex_d.npu_write32(
                            261888,
                            289,
                            column=trace_info.traced_tile_idx[0],
                            row=trace_info.traced_tile_idx[1],
                        )
                        aiex_d.npu_write32(
                            261892,
                            0,
                            column=trace_info.traced_tile_idx[0],
                            row=trace_info.traced_tile_idx[1],
                        )
                        aiex_d.npu_write32(
                            212992,
                            31232,
                            column=trace_info.traced_tile_idx[0],
                            row=trace_info.traced_tile_idx[1],
                        )
                        aiex_d.npu_writebd(
                            bd_id=Config.DMA_MAX_BDS - trace_info.packet_id,
                            buffer_length=trace_size,
                            buffer_offset=0,
                            enable_packet=1,
                            out_of_order_id=0,
                            packet_id=trace_info.packet_id,
                            packet_type=0,
                            column=trace_info.shim_tile_idx[0],
                            d0_size=0,
                            d0_stride=0,
                            d0_zero_after=0,
                            d0_zero_before=0,
                            d1_size=0,
                            d1_stride=0,
                            d1_zero_after=0,
                            d1_zero_before=0,
                            d2_size=0,
                            d2_stride=0,
                            d2_zero_after=0,
                            d2_zero_before=0,
                            burst_length=64,
                            iteration_current=0,
                            iteration_size=0,
                            iteration_stride=0,
                            lock_acq_enable=0,
                            lock_acq_id=0,
                            lock_acq_val=0,
                            lock_rel_id=0,
                            lock_rel_val=0,
                            next_bd=0,
                            row=0,
                            use_next_bd=0,
                            valid_bd=1,
                        )
                        aiex_d.npu_address_patch(
                            33554432 * trace_info.shim_tile_idx[0]
                            + 119268
                            - 32 * (trace_info.packet_id - 1),
                            len(self.global_inputs) + len(self.global_outputs),
                            0,
                        )
                        aiex_d.npu_write32(
                            119308,
                            Config.DMA_MAX_BDS - trace_info.packet_id,
                            column=trace_info.shim_tile_idx[0],
                            row=trace_info.shim_tile_idx[1],
                        )
                    if len(enabled_trace) > 0:
                        aiex_d.npu_write32(
                            212992,
                            32512,
                            column=enabled_trace[0].shim_tile_idx[0],
                            row=enabled_trace[0].shim_tile_idx[1],
                        )
                        aiex_d.npu_write32(
                            213068,
                            127,
                            column=enabled_trace[0].shim_tile_idx[0],
                            row=enabled_trace[0].shim_tile_idx[1],
                        )
                        aiex_d.npu_write32(
                            213000,
                            127,
                            column=enabled_trace[0].shim_tile_idx[0],
                            row=enabled_trace[0].shim_tile_idx[1],
                        )

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
