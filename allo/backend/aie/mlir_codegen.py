# pylint: disable=import-error, no-name-in-module, c-extension-no-member, too-many-branches, too-many-nested-blocks, redefined-variable-type, consider-using-enumerate, too-many-instance-attributes, chained-comparison, cell-var-from-loop
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import copy
from typing import Any
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
from ..._mlir.dialects import (
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
    get_aie_mlir_dtype_from_str,
    merge_token_sets,
    device_config_map,
    Argument,
    Stream,
    Config,
    string_sort_key,
    RuntimeArgs,
)
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


class CodeGenerator:
    """
    CodeGenerator is responsible for transforming Allo functions and their associated
    DTensor-based input/output mappings into AIE (AI Engine) core-compatible IR.
    It manages stream transformations, memory operations, and integrates with the
    AIE dialect of MLIR.
    """

    # pylint: disable=unsupported-binary-operation
    def __init__(
        self,
        device_type: str,
        global_tensors: dict[int, DTensor],
        top_function: allo_func_d.FuncOp,
        core_func_args: dict[str, dict[int, tuple[Argument, bool]]],
        streams: dict[str, Stream],
        virtual_computation_graph: ComputationGraph = None,
    ):
        self.device_type = device_type
        self.device_config = device_config_map[device_type]
        assert self.device_config is not None, "Unsupported device type"

        self.global_tensors: dict[int, DTensor] = global_tensors
        self.arg_slots_in_runtime_args: dict[int, tuple[int, int]] = {}
        self.module_runtime_args: list[RuntimeArgs] = []
        self.top_function = top_function
        self.core_func_args = core_func_args
        self.streams = streams
        self.virtual_computation_graph: ComputationGraph = virtual_computation_graph

        self.tile_map: dict[str, aie_d.TileOp] = {}
        self.fifo_map: dict[str, aie_d.object_fifo | tuple[aie_d.object_fifo]] = {}
        # function name (with id) -> a map from DTensor to fifo name
        self.compute_core_io: dict[str : dict[DTensor, str]] = {}
        self.external_functions: str = ""

        # ------------------------------------------------------------
        # Experimental
        # ------------------------------------------------------------
        self.mem_tile_idx = 0
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

    # pylint: disable=unsupported-binary-operation
    def preporocess_dumped_core_func(
        self,
        original_func: allo_func_d.FuncOp,
        func_args: dict[int, tuple[Argument | list[Argument], bool]],
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
                if idx in func_args:
                    sample_stream = (
                        func_args[idx][0].stream
                        if isinstance(func_args[idx][0], Argument)
                        else func_args[idx][0][0].stream
                    )
                    if sample_stream is not None:
                        new_func_inputs.append(sample_stream.allo_element_type)
                        func_inputs[idx] = sample_stream.allo_element_type
                        continue
                    if func_args[idx][0].dtensor is not None:
                        new_func_inputs.append(func_inputs[idx])
                        continue
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
                sample_stream = (
                    arg_info[0].stream
                    if isinstance(arg_info[0], Argument)
                    else arg_info[0][0].stream
                )
                if sample_stream is not None:
                    argument = new_function.arguments[idx]
                    for use_ in argument.uses:
                        op = use_.owner
                        if op.name == "allo.stream_put":
                            operands = op.operands
                            # store/copy
                            if sample_stream.is_tensor:
                                new_op = allo_memref_d.CopyOp(
                                    operands[1], operands[0], ip=InsertionPoint(op)
                                )
                            else:
                                new_op = allo_memref_d.StoreOp(
                                    operands[1], operands[0], [], ip=InsertionPoint(op)
                                )
                        elif op.name == "allo.stream_get":
                            # load/alloc
                            if sample_stream.is_tensor:
                                # replace use with alloc
                                new_op = allo_memref_d.AllocOp(
                                    sample_stream.allo_element_type,
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
        if "input_depth" in original_func.attributes:
            input_arg_depth = []
            for elem in original_func.attributes["input_depth"]:
                input_arg_depth.append(elem.value)
        else:
            input_arg_depth = None
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
            reused_fifo_info: dict[str, tuple[bool, Any]] = {}
            for i, argument in enumerate(parsed_function.arguments):
                if not i in func_args:
                    continue
                arg_info: tuple[Argument, bool] = func_args[i]
                if (
                    isinstance(arg_info[0], Argument)
                    and arg_info[0].dtensor is not None
                ):
                    nest_depth = (
                        input_arg_depth[i] if input_arg_depth is not None else 0
                    )
                    # fixme: argument.uses is unordered??
                    arg_use = list(argument.uses)
                    first_use = arg_use[-1] if len(arg_use) > 0 else None
                    if first_use is not None:
                        first_use_op = first_use.owner
                        # find parenting nest
                        while nest_depth > 0:
                            while "task_nest" not in first_use_op.parent.attributes:
                                first_use_op = first_use_op.parent
                            nest_depth -= 1
                            first_use_op = first_use_op.parent
                        while (
                            first_use_op.parent.name != "func.func"
                            and "task_nest" not in first_use_op.parent.attributes
                        ):
                            first_use_op = first_use_op.parent
                        fifo = self.fifo_map[arg_to_fifo[i].name]
                        block = first_use_op.parent.regions[0].blocks[0]
                        with aie_ir.InsertionPoint(first_use_op):
                            if arg_to_fifo[i].name in reused_fifo_info:
                                assert block == reused_fifo_info[arg_to_fifo[i].name][1]
                                fifo.release(
                                    1 if arg_info[0].dtensor.is_input else 0, 1
                                )
                            else:
                                reused_fifo_info[arg_to_fifo[i].name] = (
                                    arg_info[0].dtensor.is_input,
                                    block,
                                )
                            acquired = fifo.acquire(
                                1 if arg_info[0].dtensor.is_input else 0, 1
                            )
                            # incorrect
                            argument.replace_all_uses_with(acquired)
                else:
                    fifo_list = []
                    if isinstance(arg_info[0], Argument):
                        fifo_list.append(self.fifo_map[arg_info[0].stream.name])
                    else:
                        for stream_arg in arg_info[0]:
                            fifo_list.append(self.fifo_map[stream_arg.stream.name])
                    compact_flag = len(fifo_list) == 1 or len(set(fifo_list)) == 1
                    fifo = fifo_list[0]
                    for use_ in argument.uses:
                        op = use_.owner
                        # get loop nests
                        loop_nests = {}
                        parent = op.parent
                        while parent is not None and not isinstance(
                            parent, aie_func_d.FuncOp
                        ):
                            if (
                                "loop_name" in parent.attributes
                                and "op_name" in parent.attributes
                            ):
                                loop_nests[parent.attributes["op_name"].value] = parent
                            parent = parent.parent
                        with aie_ir.InsertionPoint(op.operation):
                            is_put, is_tensor = None, None
                            if op.name == "memref.store" or (
                                op.name == "memref.copy" and argument == op.operands[1]
                            ):  # allo.stream_put
                                is_put = True
                            elif (
                                op.name == "memref.load"
                            ):  # allo.stream_get, non-tensor
                                is_put, is_tensor = False, False
                            elif op.name == "memref.copy":  # allo.stream_get, tensor
                                is_put, is_tensor = False, True
                            else:
                                continue
                            if compact_flag:
                                if isinstance(fifo, tuple):
                                    fifo = fifo[0 if is_put else 1]
                                acquired = fifo.acquire(0 if is_put else 1, 1)
                                if is_put:
                                    op.operands[1] = acquired
                                    new_op = op.clone()  # no use, no need to replace
                                else:
                                    op.operands[0] = acquired
                                    new_op = op.clone()
                                    if not is_tensor:
                                        for old, new in zip(op.results, new_op.results):
                                            old.replace_all_uses_with(new)
                                fifo.release(0 if is_put else 1, 1)
                            else:
                                assert len(loop_nests) == 1, "To be implemented..."
                                loop_name = list(loop_nests.keys())[0]
                                cases = []
                                case_val = []
                                for fifo, stream_arg in zip(fifo_list, arg_info[0]):
                                    if is_put:
                                        cases.append(
                                            stream_arg.stream.src_related_iter_info[
                                                loop_name
                                            ]
                                        )
                                    else:
                                        cases.append(
                                            stream_arg.stream.dst_related_iter_info[
                                                loop_name
                                            ]
                                        )
                                    if isinstance(fifo, tuple):
                                        case_val.append(fifo[0 if is_put else 1])
                                    else:
                                        case_val.append(fifo)
                                switch_op = aie_scf_d.IndexSwitchOp(
                                    [op.operands[1].type],
                                    loop_nests[loop_name]
                                    .regions[0]
                                    .blocks[0]
                                    .arguments[0],
                                    cases[1:],
                                    len(cases[1:]),
                                )
                                cnt = 0
                                for region in switch_op.caseRegions:
                                    block = region.blocks.append()
                                    with aie_ir.InsertionPoint(block):
                                        acquired = case_val[cnt].acquire(
                                            0 if is_put else 1, 1
                                        )
                                        aie_scf_d.YieldOp([acquired])
                                        cnt += 1
                                if is_put:
                                    op.operands[1] = switch_op.result
                                    new_op = op.clone()  # no use, no need to replace
                                else:
                                    op.operands[0] = switch_op.result
                                    new_op = op.clone()
                                    if not is_tensor:
                                        for old, new in zip(op.results, new_op.results):
                                            old.replace_all_uses_with(new)
                                switch_op = aie_scf_d.IndexSwitchOp(
                                    [],
                                    loop_nests[loop_name]
                                    .regions[0]
                                    .blocks[0]
                                    .arguments[0],
                                    cases[1:],
                                    len(cases[1:]),
                                )
                                cnt = 0
                                for region in switch_op.caseRegions:
                                    block = region.blocks.append()
                                    with aie_ir.InsertionPoint(block):
                                        case_val[cnt].release(0 if is_put else 1, 1)
                                        aie_scf_d.YieldOp([])
                                        cnt += 1
                            op.erase()

            for fifo_name, (is_input, region) in reused_fifo_info.items():
                with aie_ir.InsertionPoint.at_block_terminator(region):
                    self.fifo_map[fifo_name].release(1 if is_input else 0, 1)

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
        stride: list[int]

        def transfer_pattern(self):
            offset, size, stride = (
                list(self.offset),
                list(self.size),
                list(self.stride),
            )
            if size[0] > 1 and size[1] == 1:
                offset[0], offset[1] = offset[1], offset[0]
                size[0], size[1] = size[1], size[0]
                stride[0], stride[1] = stride[1], stride[0]
            return offset, size, stride

        def print(self):
            print(
                self.token,
                f"[{self.start_time, self.end_time}]",
                self.dtensor.global_id,
                self.offset,
                self.size,
            )

    class DMATaskWithSameToken:
        def __init__(self, task: "CodeGenerator.GlobalIODMATask"):
            self.start_time: int = task.start_time
            self.end_time: int = task.end_time
            self.tasks: list[CodeGenerator.GlobalIODMATask] = [task]
            self.related_pes: set[str] = set()
            for interfaces in task.io_port.connect_interface:
                for interface in interfaces.interface_list:
                    self.related_pes.add(interface.pe)

        def add(self, task: "CodeGenerator.GlobalIODMATask"):
            self.start_time = min(self.start_time, task.start_time)
            self.end_time = max(self.end_time, task.end_time)
            self.tasks.append(task)
            for interfaces in task.io_port.connect_interface:
                for interface in interfaces.interface_list:
                    self.related_pes.add(interface.pe)

        def print(self):
            print("-+-")
            print(self.related_pes)
            for task in self.tasks:
                task.print()
            print("---")

    def map_data_transfer(self) -> dict[str, dict[int, FIFO]]:
        """
        Construct data transfer path from external memory to each (logical) compute tile.

        TODO: may have influence on DMA scheduling
        """

        def partition(size: Size4D) -> tuple[int, Size4D]:
            """
            Partition the dma task into multiple sub-tasks.
            """
            # find the first none-1 dim
            for dim in range(4):
                if size.get_dim_size(dim) > 1:
                    break
            if dim >= 3:
                raise ValueError(f"Fail to partition {size}")
            size_part = size.copy()
            partition_size = size.get_dim_size(dim) - 1
            size_part.set_dim_size(dim, partition_size)
            return dim, size_part

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
            # cyclic graph
            if len(tagged_nodes) == 0:
                for node in dependencies.keys():
                    node_order_tag[node] = tag
                break
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
        global_dtensor = self.global_tensors
        global_tile_to_func: dict[int, DTensorTileGroup] = {
            i: DTensorTileGroup("") for i in self.global_tensors.keys()
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
                self,
                other: "MulticastInterface",
                current_size: Size4D,
                contiguous_dim: int,
            ) -> tuple[Size4D, int]:
                for interface in self.interface_list:
                    if interface in other.interface_list:
                        # TODO: can be relaxed
                        return None, None
                if self.sample_interface.layout != other.sample_interface.layout:
                    return None, None
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
                        size_contiguous_dim = None
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
                                    idx_diff = (
                                        other_flattened_idx - sample_flattened_idx
                                    )
                                    for i in range(4):
                                        if idx_diff >= outer_stride[i]:
                                            if idx_diff % outer_stride[i] != 0:
                                                match_flag = False
                                                break
                                            idx_diff //= outer_stride[i]
                                            if idx_diff == 1:
                                                size_contiguous_dim = i
                                                break
                                    if (
                                        contiguous_dim is not None
                                        and contiguous_dim != size_contiguous_dim
                                    ):
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
                            return None, None
                    self.tokens.add(tuple(new_token_list))
                    return Size4D.from_list(new_size_list), size_contiguous_dim
                return None, None

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

            def __init__(self, interface: MulticastInterface = None):
                if interface is not None:
                    self.layout = interface.sample_interface.layout
                    self.total_size: Size4D = Size4D(1, 1, 1, 1)
                    self.interface_list: list[MulticastInterface] = [interface]
                else:
                    self.layout = None
                    self.total_size: Size4D = None
                    self.interface_list: list[MulticastInterface] = []
                self.contiguous_dim = None

            def append(self, other: MulticastInterface) -> bool:
                sample = self.interface_list[-1]
                updated_size, contiguous_dim = sample._contiguous_data_transfer(
                    other, self.total_size, self.contiguous_dim
                )
                if updated_size is None:
                    return False
                if (
                    self.contiguous_dim is not None
                    and contiguous_dim != self.contiguous_dim
                ):
                    return False
                self.contiguous_dim = contiguous_dim
                self.interface_list.append(other)
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
            if str(dtype) == "i4":
                tile_total_size //= 2
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
                    col_id=len(self.used_mem_tiles),
                    row_id=1,
                )
                self.used_mem_tiles.append(assigned_mem_tile)
                self.mem_tile_idx = len(self.used_mem_tiles)
            else:
                # Attempt to use an existing memory tile
                for offset in range(len(self.used_mem_tiles)):
                    mem_tile = self.used_mem_tiles[
                        (self.mem_tile_idx + offset) % len(self.used_mem_tiles)
                    ]
                    if (
                        len(mem_tile.send_ports) + send_need <= Config.MEM_MAX_SEND
                        and len(mem_tile.recv_ports) + recv_need <= Config.MEM_MAX_RECV
                    ):
                        assigned_mem_tile = mem_tile
                        self.mem_tile_idx = (self.mem_tile_idx + offset + 1) % len(
                            self.used_mem_tiles
                        )
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
                    col_id=len(self.used_shim_tiles),
                    row_id=0,
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
            tile_dtype: str,
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
        global_dma_tasks: dict[int, list[ContiguousInterface]] = {}
        for idx, dtensor_tile_group in global_tile_to_func.items():
            dtensor = self.global_tensors[idx]
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
            coalesced_access_pattern, _, coalesced_multicast_interfaces = (
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
                                multicast_interface
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
            global_dma_tasks[idx] = contiguous_interfaces

        # ####################
        # # HACK: an aggressive strategy to fully utilize interface ports (may be problematic)
        # ####################
        if os.getenv("ENABLE_AGGRESSIVE_PORT_UTILIZATION_PATCH") == "1":
            global_input_num, global_output_num = 0, 0
            for idx, contiguous_interfaces in global_dma_tasks.items():
                if self.global_tensors[idx].is_input:
                    global_input_num += len(contiguous_interfaces)
                else:
                    global_output_num += len(contiguous_interfaces)
            for idx, contiguous_interfaces in global_dma_tasks.items():
                # fixme: MAX_SHIM_TILES (how to select better factor?) adjust base on independent workload?
                FACTOR = int(os.getenv("FACTOR", MAX_SHIM_TILES))
                if len(contiguous_interfaces) == 1:
                    factor = FACTOR
                    while factor > 1:
                        if len(contiguous_interfaces[0].interface_list) % factor == 0:
                            if (
                                factor
                                - len(contiguous_interfaces)
                                + (
                                    global_input_num
                                    if self.global_tensors[idx].is_input
                                    else global_output_num
                                )
                                <= 2 * FACTOR
                            ):
                                break
                        factor >>= 1
                    if factor > 1:
                        contiguous_interface: ContiguousInterface = (
                            contiguous_interfaces[0]
                        )
                        contiguous_interfaces.clear()
                        total_tile = contiguous_interface.total_size.get_total_size()
                        slice_size = total_tile // factor
                        for i in range(factor):
                            new_contiguous_interface = ContiguousInterface()
                            new_contiguous_interface.contiguous_dim = (
                                contiguous_interface.contiguous_dim
                            )
                            new_contiguous_interface.layout = (
                                contiguous_interface.layout
                            )
                            new_contiguous_interface.total_size = (
                                contiguous_interface.total_size.get_k_slice(slice_size)
                            )
                            new_contiguous_interface.interface_list.extend(
                                contiguous_interface.interface_list[
                                    i * slice_size : (i + 1) * slice_size
                                ]
                            )
                            contiguous_interfaces.append(new_contiguous_interface)

                        if self.global_tensors[idx].is_input:
                            global_input_num -= 1
                            global_input_num += len(contiguous_interfaces)
                        else:
                            global_output_num -= 1
                            global_output_num += len(contiguous_interfaces)
        # ####################

        for idx, contiguous_interfaces in global_dma_tasks.items():
            self.mem_tile_idx = 0
            dtensor = self.global_tensors[idx]
            tile_shape = list(dtensor.size)
            for i in dtensor.shared_dims:
                tile_shape[i] = 1
            tile_size = Size4D.from_list(tile_shape)
            for contiguous_interface in contiguous_interfaces:
                interface_list: list[MulticastInterface] = (
                    contiguous_interface.interface_list
                )
                size = contiguous_interface.total_size
                transfer_layout = contiguous_interface.layout

                def transfer(size_: Size4D, interface_list_: list[MulticastInterface]):
                    dim_ = None
                    while size_.get_total_size() != 0:
                        if assign_tiles(
                            interface_list_,
                            size_,
                            tile_size,
                            dtensor.is_input,
                            dtensor.dtype,
                            dtensor.type_as_param,
                        ):
                            break
                        size_cp = size_.copy()
                        # keep partitioning until success
                        while True:
                            partitioned_dim, partitioned_size = partition(
                                size_cp
                            )  # partition too much, drop dim
                            partitioned_interface_list = interface_list_[
                                : partitioned_size.get_total_size()
                            ]
                            if dim_ is None:
                                dim_ = partitioned_dim
                            elif dim_ != partitioned_dim:
                                transfer(
                                    size_cp, interface_list_[: size_cp.get_total_size()]
                                )
                                partitioned_size = size_cp
                                break
                            if assign_tiles(
                                partitioned_interface_list,
                                partitioned_size,
                                tile_size,
                                dtensor.is_input,
                                dtensor.dtype,
                                dtensor.type_as_param,
                            ):
                                break
                            size_cp = partitioned_size
                        size_ = Size4D.subtract(size_, partitioned_size)
                        inc = partitioned_size.get_total_size()
                        interface_list_ = interface_list_[inc:]

                transfer(size, interface_list)

        # insert placeholder for mem/shim tiles
        while len(self.used_mem_tiles) < MAX_MEM_TILES:
            assigned_mem_tile = SwitchNode(
                name=f"{len(self.used_mem_tiles)}_mem_tile",
                send_port_num=Config.MEM_MAX_SEND,
                recv_port_num=Config.MEM_MAX_RECV,
                col_id=len(self.used_mem_tiles),
                row_id=1,
            )
            self.used_mem_tiles.append(assigned_mem_tile)
        while len(self.used_shim_tiles) < MAX_SHIM_TILES:
            assigned_shim_tile = SwitchNode(
                name=f"{len(self.used_shim_tiles)}_shim_tile",
                send_port_num=Config.SHIM_MAX_SEND,
                recv_port_num=Config.SHIM_MAX_RECV,
                col_id=len(self.used_shim_tiles),
                row_id=0,
            )
            self.used_shim_tiles.append(assigned_shim_tile)

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
                    dtensor_ = self.global_tensors[live_tensor_tile.tile.dtensor_id]
                    size = list(io_port.size)
                    offset = dtensor_.offset_map[
                        live_tensor_tile.tile.tensor_tile_label
                    ].to_list()
                    stride = list(io_port.stride)
                    # fixme: patch only, find a robust way for higher dimensions??
                    if size[0] > 1 and size[1] == 1:
                        size[0], size[1] = size[1], size[0]
                        offset[0], offset[1] = offset[1], offset[0]
                        stride[0], stride[1] = stride[1], stride[0]
                    self.global_dma_trough_port.append(
                        CodeGenerator.GlobalIODMATask(
                            token=token_map[live_tensor_tile.token],
                            start_time=live_tensor_tile.first_use
                            + Config.GLOBAL_CODE_OFFSET * io_port.order_tag,
                            end_time=live_tensor_tile.last_use
                            + Config.GLOBAL_CODE_OFFSET * io_port.order_tag,
                            io_port=io_port,
                            dtensor=dtensor_,
                            size=size,
                            offset=offset,
                            stride=stride,
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

    def map_core_func_to_physical_tiles(
        self, layout_col_id_hint: dict[str, int]
    ) -> dict[str, tuple[int, int]]:
        """
        Map the core functions to physical tiles.
        TODO:
            - mapping strategies should be selected by cost
            - careful with nodes with multiple inputs/outputs
              (if ports are exceeded, we should try to assign them to adjacent compute tiles to share local memory)
        """
        core_func_mapping: dict[str, tuple[int, int]] = {}
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
                    grouped_nodes.get(connection[1]),
                    grouped_nodes.get(connection[2]),
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
            linked_nodes: list[NodeDeque] = []
            single_nodes: list[str] = []
            for node_deque in sorted_values:
                if node_deque in assigned:
                    continue
                assigned.add(node_deque)
                if len(node_deque.nodes) == 1:
                    single_nodes.append(node_deque.nodes[0])
                else:
                    linked_nodes.append(node_deque)
            for deque in linked_nodes:
                head = deque.nodes[0]
                while tile_used[traverse_idx // max_col][traverse_idx % max_col]:
                    traverse_idx += 1
                    if traverse_idx >= max_row * max_col:
                        raise ValueError("Too many nodes")
                col_idx = traverse_idx % max_col
                row_idx = traverse_idx // max_col
                core_func_mapping[head] = (row_idx, col_idx)
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
                    core_func_mapping[node] = (row_idx, col_idx)
                    tile_used[row_idx][col_idx] = True
            for node in single_nodes:
                for col_idx in range(layout_col_id_hint[node], max_col):
                    row_idx = 0
                    while row_idx < max_row and tile_used[row_idx][col_idx]:
                        row_idx += 1
                    if row_idx < max_row:
                        core_func_mapping[node] = (row_idx, col_idx)
                        tile_used[row_idx][col_idx] = True
                        break
                if node not in core_func_mapping:
                    for col_idx in range(0, layout_col_id_hint[node]):
                        row_idx = 0
                        while row_idx < max_row and tile_used[row_idx][col_idx]:
                            row_idx += 1
                        if row_idx < max_row:
                            core_func_mapping[node] = (row_idx, col_idx)
                            tile_used[row_idx][col_idx] = True
                            break
                if node not in core_func_mapping:
                    raise ValueError(f"Fail to map {node}")
            if os.getenv("VERBOSE") == "1":
                print("<<< Mapping >>>")
                for node, (row, col) in core_func_mapping.items():
                    print(f"{node}: ({row}, {col})")
                print()
            return core_func_mapping
        raise ValueError("To be implemented")

    # ############################################################
    # AIE Code Generation
    # ############################################################
    def aie_codegen(
        self,
        core_funcs: list[allo_func_d.FuncOp],
        external_funcs: list[allo_func_d.FuncOp],
        linked_external_cc: dict[str, int],
        trace: list[tuple[str, tuple[int, ...]]],
        trace_size: int,
    ) -> aie_ir.Module:
        # mapping to physical/logical
        # TODO: co-designed mapping to different types of tiles
        arg_to_fifo = self.map_data_transfer()
        core_func_connected_mem_tile: dict[str, dict[str, int]] = {}
        for func_name in arg_to_fifo.keys():
            core_func_connected_mem_tile[func_name] = {
                mem_tile.name: 0 for mem_tile in self.used_mem_tiles
            }
        for func_name, fifo_dict in arg_to_fifo.items():
            for fifo in fifo_dict.values():
                if fifo.src == func_name:
                    core_func_connected_mem_tile[func_name][fifo.dst[0]] += 1
                else:
                    core_func_connected_mem_tile[func_name][fifo.src] += 1
        layout_col_id_hint: dict[str, str] = {}
        for func_name, connections in core_func_connected_mem_tile.items():
            heaviest: int = -1
            heaviest_workload = -1
            for mem_tile_name, workload in connections.items():
                if heaviest_workload < workload:
                    heaviest_workload = workload
                    heaviest = int(mem_tile_name[0])  # fixme
            layout_col_id_hint[func_name] = heaviest
        core_function_mapping = self.map_core_func_to_physical_tiles(layout_col_id_hint)

        # traced tile index
        traced_logical_tile = set()
        available_shim_for_trace: str = None
        if trace is not None:
            virtual_to_logical = {}
            for node_name, node in self.virtual_computation_graph.nodes.items():
                for virtual_name in node.meta_data.df_kernels:
                    virtual_to_logical[virtual_name] = node_name
            for traced_tile in trace:
                func_name = (
                    traced_tile[0] + f"_{'_'.join([str(x) for x in traced_tile[1]])}"
                )
                assert func_name in virtual_to_logical
                traced_logical_tile.add(virtual_to_logical[func_name])

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
                for shim_tile in self.used_shim_tiles:
                    self.tile_map[shim_tile.name] = aie_d.TileOp(
                        col=shim_tile.col_id, row=shim_tile.row_id
                    )
                    if (
                        available_shim_for_trace is None
                        and len(shim_tile.send_ports) == 0
                    ):
                        available_shim_for_trace = shim_tile.name
                # mem tiles
                for mem_tile in self.used_mem_tiles:
                    self.tile_map[mem_tile.name] = aie_d.TileOp(
                        col=mem_tile.col_id, row=mem_tile.row_id
                    )
                # compute tiles
                for func_name, (row, col) in core_function_mapping.items():
                    self.tile_map[func_name] = aie_d.TileOp(col=col, row=row + 2)
                # define fifos
                # - stream fifos: compute <-> compute
                for stream_name, stream in self.streams.items():
                    dimensions_to_stream = stream.get_dimensions_to_stream()
                    if len(dimensions_to_stream) <= Config.COMP_COMP_DMA_TRANSFORM_DIM:
                        self.fifo_map[stream_name] = aie_d.object_fifo(
                            stream_name,
                            self.tile_map[stream.src],
                            self.tile_map[stream.dst],
                            depth=stream.type.depth,
                            datatype=aie_ir.MemRefType.get(
                                stream.type.shape,
                                get_aie_mlir_dtype_from_str(str(stream.type.dtype)),
                            ),
                            dimensionsToStream=dimensions_to_stream,
                        )
                    elif len(dimensions_to_stream) <= Config.MEM_COMP_DMA_TRANSFORM_DIM:
                        # prioritize the mem tile in the same column
                        switch_mem = self.used_mem_tiles[
                            core_function_mapping[stream.src][1]
                        ]
                        if (
                            len(switch_mem.recv_ports) == switch_mem.max_recv
                            or len(switch_mem.send_ports) == switch_mem.max_send
                        ):
                            switch_mem = None
                            for mem_tile in self.used_mem_tiles:
                                if (
                                    len(mem_tile.recv_ports) < mem_tile.max_recv
                                    and len(mem_tile.send_ports) <= mem_tile.max_send
                                ):
                                    switch_mem = mem_tile
                                    break
                        stream_src = aie_d.object_fifo(
                            stream_name + "_src",
                            self.tile_map[stream.src],
                            self.tile_map[switch_mem.name],
                            depth=stream.type.depth,
                            datatype=aie_ir.MemRefType.get(
                                stream.type.shape,
                                get_aie_mlir_dtype_from_str(str(stream.type.dtype)),
                            ),
                        )
                        stream_dst = aie_d.object_fifo(
                            stream_name + "_dst",
                            self.tile_map[switch_mem.name],
                            self.tile_map[stream.dst],
                            depth=stream.type.depth,
                            datatype=aie_ir.MemRefType.get(
                                stream.type.shape,
                                get_aie_mlir_dtype_from_str(str(stream.type.dtype)),
                            ),
                            dimensionsToStream=dimensions_to_stream,
                        )
                        switch_mem.send_ports.append(None)
                        switch_mem.recv_ports.append(None)
                        self.fifo_map[stream_name] = (stream_src, stream_dst)
                        aie_d.object_fifo_link([stream_src], [stream_dst], [], [])
                    else:
                        raise ValueError(
                            "layout transformation cannot be achieved on DMA"
                        )
                # - io fifos: shim <-> mem <-> compute
                for dma_fifo in self.fifo_manager.fifos:
                    assert (
                        len(dma_fifo.dimensions_to_stream)
                        <= Config.MEM_COMP_DMA_TRANSFORM_DIM
                    )
                    self.fifo_map[dma_fifo.name] = aie_d.object_fifo(
                        dma_fifo.name,
                        self.tile_map[dma_fifo.src],
                        [self.tile_map[node] for node in dma_fifo.dst],
                        depth=dma_fifo.depth,
                        datatype=aie_ir.MemRefType.get(
                            dma_fifo.data_shape,
                            get_aie_mlir_dtype_from_str(str(dma_fifo.dtype)),
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
                    used_external_kernel = self.virtual_computation_graph.nodes[
                        func_name
                    ].meta_data.used_external_kernel
                    func_core = aie_d.Core(
                        tile=self.tile_map[func_name],
                        link_with=(
                            f"external{linked_external_cc[func_name]}.o"
                            if len(used_external_kernel) > 0
                            else None
                        ),
                    )
                    if self.global_ip is None:
                        self.global_ip = aie_ir.InsertionPoint(func_core)
                    self.build_core_function(
                        func_core,
                        func,
                        self.core_func_args[func_name],
                        arg_to_fifo[func_name],
                    )

                @dataclass
                class TraceInfo:
                    traced_tile_idx: tuple[int]
                    shim_tile_idx: tuple[int]
                    packet_id: int

                enabled_trace: list[TraceInfo] = []

                trace_transfer_shim_tile = None
                if available_shim_for_trace is not None:
                    trace_transfer_shim_tile = self.tile_map[available_shim_for_trace]
                elif len(self.used_shim_tiles) < self.device_config["shim_tile_num"]:
                    trace_transfer_shim_tile = aie_d.TileOp(
                        col=len(self.used_shim_tiles), row=0
                    )
                packet_id = 0
                if (
                    len(traced_logical_tile) > 0
                    and trace_transfer_shim_tile is not None
                ):
                    max_pack_num = (
                        Config.TRACE_MAX_NUM
                        if available_shim_for_trace is not None
                        else Config.DMA_MAX_BDS
                    )
                    # TODO: Trace, reserve how many bds for tracing
                    for traced_tile in traced_logical_tile:
                        packet_id += 1
                        if packet_id > max_pack_num:
                            break
                        compute_tile = self.tile_map[traced_tile]
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
                global_tensor_types: dict[
                    tuple[str, bool], tuple[RuntimeArgs, list[int]]
                ] = {}
                for i in range(len(self.global_tensors)):
                    arg = self.global_tensors[i]
                    tensor_type = (arg.dtype, arg.is_input)
                    if tensor_type not in global_tensor_types:
                        global_tensor_types[tensor_type] = (
                            RuntimeArgs(str(arg.dtype), arg.is_input),
                            [],
                        )
                    global_tensor_types[tensor_type][0].global_tensors.append(i)
                assert (
                    len(global_tensor_types) <= Config.MAX_IO_BUFFER
                ), "unable to construct buffers for arguments"
                original_runtime_args = list(global_tensor_types.values())
                for i in range(Config.MAX_IO_BUFFER):
                    self.module_runtime_args.append(
                        RuntimeArgs(
                            str(
                                original_runtime_args[i % len(original_runtime_args)][
                                    0
                                ].raw_dtype
                            ),
                            original_runtime_args[i % len(original_runtime_args)][
                                0
                            ].is_input,
                        )
                    )
                    original_runtime_args[i % len(original_runtime_args)][1].append(i)
                for runtime_arg_idxs, arg_slots in original_runtime_args:
                    for i, arg_idx in enumerate(runtime_arg_idxs.global_tensors):
                        runtime_arg = self.module_runtime_args[
                            arg_slots[i % len(arg_slots)]
                        ]
                        runtime_arg.global_tensors.append(arg_idx)
                        self.arg_slots_in_runtime_args[arg_idx] = (
                            arg_slots[i % len(arg_slots)],
                            runtime_arg.current_size,
                        )
                        runtime_arg.inc_size(self.global_tensors[arg_idx].shape)

                runtime_seq = aiex_d.RuntimeSequenceOp()
                runtime_args = []
                for runtime_arg in self.module_runtime_args:
                    if len(runtime_arg.global_tensors) == 0:
                        continue
                    runtime_args.append(
                        aie_ir.MemRefType.get(
                            [runtime_arg.current_size],
                            get_aie_mlir_dtype_from_str(str(runtime_arg.dtype)),
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
                independent_dma_task_groups: list[
                    tuple[set[int], list[CodeGenerator.DMATaskWithSameToken]]
                ] = []
                for task_group in dma_task_groups.values():
                    inserted = False
                    for independent_dma_task_group in independent_dma_task_groups:
                        if not independent_dma_task_group[0].isdisjoint(
                            task_group.related_pes
                        ):
                            independent_dma_task_group[0].update(task_group.related_pes)
                            independent_dma_task_group[1].append(task_group)
                            inserted = True
                            break
                    if not inserted:
                        independent_dma_task_groups.append(
                            (set(task_group.related_pes), [task_group])
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
                            len(self.global_tensors),
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

                    # data with same token should be transferred together
                    # fixme: if execution fails with runtime_error, possibly because the transfer order leads to 'deadlock'
                    max_dma_task_group_size = 0
                    for independent_dma_task_group in independent_dma_task_groups:
                        independent_dma_task_group[1].sort(key=lambda x: x.start_time)
                        max_dma_task_group_size = max(
                            max_dma_task_group_size, len(independent_dma_task_group[1])
                        )
                    coalesced_tasks_list: list[list[CodeGenerator.GlobalIODMATask]] = []
                    for i in range(max_dma_task_group_size):
                        tasks = []
                        for independent_dma_task_group in independent_dma_task_groups:
                            if i < len(independent_dma_task_group[1]):
                                tasks.extend(independent_dma_task_group[1][i].tasks)
                        tasks.sort(key=lambda x: x.start_time)
                        fifo_to_tasks: dict[
                            str, list[CodeGenerator.GlobalIODMATask]
                        ] = defaultdict(list)
                        for global_dma in tasks:
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
                                            incoming_offset = Offset4D(
                                                tasks[right].offset[0],
                                                tasks[right].offset[1],
                                                tasks[right].offset[2],
                                                tasks[right].offset[3],
                                            )
                                            idx = current_offset.check_next_offset(
                                                incoming_offset
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
                                                current_offset = incoming_offset
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
                                        tasks[left].stride,
                                    )
                                    coalesced_fifo_to_tasks[fifo].append(coalesced_task)
                                    left = right
                            fifo_to_tasks = coalesced_fifo_to_tasks
                        coalesced_tasks: list[CodeGenerator.GlobalIODMATask] = []
                        for tasks in fifo_to_tasks.values():
                            coalesced_tasks.extend(tasks)
                        coalesced_tasks_list.append(coalesced_tasks)

                    tasks_idx_left = 0

                    @dataclass
                    class DMAMemcpyGroup:
                        dma_tasks: list[tuple[list[int], list[int], list[int]]]
                        diff: list[int]
                        bd_id: int
                        dtensor_global_id: bool
                        used_shim: str
                        max_task_id: int

                    guards = {}
                    async_dma_tasks: list[DMAMemcpyGroup] = []
                    dma_bd_workload: dict[str, set[int]] = {
                        shim_tile.name: set(range(Config.DMA_MAX_BDS))
                        for shim_tile in self.used_shim_tiles
                    }
                    if available_shim_for_trace is not None:
                        dma_bd_workload[available_shim_for_trace] = set(
                            range(Config.DMA_MAX_BDS - packet_id)
                        )
                    while tasks_idx_left < len(coalesced_tasks_list):
                        overload_flag = False
                        fifo_dma_tasks: dict[str, list[DMAMemcpyGroup]] = {}
                        tasks_idx_right = tasks_idx_left
                        while tasks_idx_right < len(coalesced_tasks_list):
                            updated_fifo_dma_tasks: dict[str, list[DMAMemcpyGroup]] = (
                                copy.deepcopy(fifo_dma_tasks)
                            )
                            occupied_bd_id = {
                                shim_tile.name: set()
                                for shim_tile in self.used_shim_tiles
                            }
                            for global_dma in coalesced_tasks_list[tasks_idx_right]:
                                offset, size, stride = global_dma.transfer_pattern()
                                if (
                                    global_dma.io_port.fifo.name
                                    not in updated_fifo_dma_tasks
                                ):
                                    updated_fifo_dma_tasks[
                                        global_dma.io_port.fifo.name
                                    ] = []
                                # else:
                                #     prev_task: DMAMemcpyGroup = updated_fifo_dma_tasks[
                                #         global_dma.io_port.fifo.name
                                #     ][-1]
                                # the same global tensor must be tiled in the same way
                                # if (
                                #     global_dma.dtensor.global_id
                                #     == prev_task.dtensor_global_id
                                #     and size[0] == 1
                                # ):
                                #     diff = [
                                #         x - y
                                #         for x, y in zip(
                                #             offset,
                                #             prev_task.dma_tasks[-1][0],
                                #         )
                                #     ]
                                #     # fixme: can be relaxed
                                #     if prev_task.diff is None and (
                                #         all(x >= 0 for x in diff)
                                #         and sum(1 for x in diff if x != 0) <= 1
                                #     ):
                                #         prev_task.dma_tasks.append(
                                #             (offset, size, stride)
                                #         )
                                #         prev_task.diff = diff
                                #         prev_task.max_task_id = tasks_idx_right
                                #         continue
                                #     if prev_task.diff == diff:
                                #         prev_task.dma_tasks.append(
                                #             (offset, size, stride)
                                #         )
                                #         prev_task.max_task_id = tasks_idx_right
                                #         continue

                                used_shim = (
                                    global_dma.io_port.fifo.src
                                    if global_dma.io_port.is_input
                                    else global_dma.io_port.fifo.dst[0]
                                )
                                if (
                                    len(dma_bd_workload[used_shim]) == 0
                                    or len(
                                        updated_fifo_dma_tasks[
                                            global_dma.io_port.fifo.name
                                        ]
                                    )
                                    == 4  # fixme: seems that transferring too much with the same fifo is invalid (magic number)
                                ):
                                    overload_flag = True
                                    break
                                bd_id = dma_bd_workload[used_shim].pop()
                                occupied_bd_id[used_shim].add(bd_id)
                                updated_fifo_dma_tasks[
                                    global_dma.io_port.fifo.name
                                ].append(
                                    DMAMemcpyGroup(
                                        [(offset, size, stride)],
                                        None,
                                        bd_id,
                                        global_dma.dtensor.global_id,
                                        used_shim,
                                        tasks_idx_right,
                                    )
                                )
                            if overload_flag:
                                for (
                                    shim_name,
                                    occupied_bd_ids,
                                ) in occupied_bd_id.items():
                                    dma_bd_workload[shim_name].update(occupied_bd_ids)
                                break
                            tasks_idx_right += 1
                            fifo_dma_tasks = updated_fifo_dma_tasks
                        # launch tasks in fifo_tasks and wait
                        for fifo_name, fifo_infos in fifo_dma_tasks.items():
                            for fifo_info in fifo_infos:
                                task_list = fifo_info.dma_tasks
                                offset, size, stride = task_list[0]
                                total_offset = 0
                                for i in range(4):
                                    total_offset += offset[i] * stride[i]
                                if fifo_info.diff is not None:
                                    size[0] = len(task_list)
                                    stride_0 = 0
                                    for i in range(4):
                                        stride_0 += fifo_info.diff[i] * stride[i]
                                    stride[0] = stride_0
                                offsets = (
                                    [0, 0, 0, total_offset]
                                    if fifo_info.diff is not None
                                    else offset
                                )
                                offsets[-1] += int(
                                    self.arg_slots_in_runtime_args[
                                        fifo_info.dtensor_global_id
                                    ][1]
                                )
                                if (
                                    str(
                                        self.global_tensors[
                                            fifo_info.dtensor_global_id
                                        ].dtype
                                    )
                                    == "i4"
                                ):
                                    offsets[-1] //= 2
                                    size[-1] //= 2
                                    for i in range(3):
                                        stride[i] //= 2
                                aiex_d.NpuDmaMemcpyNd(
                                    metadata=self.fifo_map[fifo_name],
                                    bd_id=fifo_info.bd_id,
                                    mem=runtime_seq_entry_block.arguments[
                                        self.arg_slots_in_runtime_args[
                                            fifo_info.dtensor_global_id
                                        ][0]
                                    ],
                                    offsets=offsets,
                                    sizes=size,
                                    strides=stride,
                                    issue_token=True,
                                )
                                if not self.global_tensors[
                                    fifo_info.dtensor_global_id
                                ].is_input:
                                    if fifo_info.max_task_id not in guards:
                                        guards[fifo_info.max_task_id] = []
                                    guards[fifo_info.max_task_id].append(
                                        self.fifo_map[fifo_name]
                                    )
                            async_dma_tasks.extend(fifo_infos)
                        tasks_idx_left = tasks_idx_right
                        output_dma_tasks_id = min(guards.keys())
                        output_dma_tasks = guards.pop(output_dma_tasks_id)
                        for launched_fifo in output_dma_tasks:
                            aiex_d.dma_wait(launched_fifo)
                        updated_async_dma_tasks = []
                        for async_dma_task in async_dma_tasks:
                            if async_dma_task.max_task_id <= output_dma_tasks_id:
                                dma_bd_workload[async_dma_task.used_shim].add(
                                    async_dma_task.bd_id
                                )
                            else:
                                updated_async_dma_tasks.append(async_dma_task)
                        async_dma_tasks = updated_async_dma_tasks
                    for output_dma_tasks in guards.values():
                        for launched_fifo in output_dma_tasks:
                            aiex_d.dma_wait(launched_fifo)
                    aie_d.EndOp()

        return self.aie_module, self.module_runtime_args
