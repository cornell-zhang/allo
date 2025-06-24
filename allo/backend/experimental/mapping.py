# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from dataclasses import dataclass
from collections import defaultdict, Counter
import allo._mlir._mlir_libs._mlir as allo_ir
from ..._mlir.ir import InsertionPoint, FunctionType, Value, UnitAttr
from ..._mlir.dialects import func as func_d, allo as allo_d
from .utils import (
    Argument,
    parse_kernel_name,
    Stream,
    get_df_kernels,
    Config,
)


# ############################################################
# Memory
# ############################################################
@dataclass
class DTensorTile:
    dtensor_id: int
    tensor_tile_label: str

    def __hash__(self):
        return hash((self.dtensor_id, self.tensor_tile_label))

    def __eq__(self, other):
        return (
            self.dtensor_id == other.dtensor_id
            and self.tensor_tile_label == other.tensor_tile_label
        )

    def __str__(self):
        return f"id[{self.dtensor_id}] ({self.tensor_tile_label})"

    def __repr__(self):
        return self.__str__()


@dataclass(frozen=True)
class PEInterface:
    pe: str
    interface_idx: int

    def __hash__(self):
        return hash((self.pe, self.interface_idx))

    def __eq__(self, other):
        return self.pe == other.pe and self.interface_idx == other.interface_idx

    def __str__(self):
        return f"{self.pe} ({self.interface_idx})"

    def __repr__(self):
        return self.__str__()


class DTensorTileGroup:
    """
    DTensor tiles -> PEs (functions) using the same DTensor tile.
    """

    def __init__(self, order_tag: str):
        self.order_tag = order_tag
        self.dtensor_tile_to_pe_interfaces: dict[DTensorTile, list[PEInterface]] = (
            defaultdict(list)
        )

    def add_tensor_tile(self, tile: DTensorTile, pe: str, interface_idx: int):
        self.dtensor_tile_to_pe_interfaces[tile].append(
            PEInterface(pe=pe, interface_idx=interface_idx)
        )

    def print(self):
        for tile, pes in self.dtensor_tile_to_pe_interfaces.items():
            print(f"{tile}: {pes}")


class OrderedDTensorTileGroup:
    """
    order_tag -> DTensorTileGroup

    `order_tag` is useful to determine the correct (deadlock-free) order of tile transfer.
    """

    def __init__(self):
        self.dtensor_tile_groups: dict[str, DTensorTileGroup] = {}

    def add_tensor_tile(
        self, tile: DTensorTile, order_tag: str, pe: str, interface_idx: int
    ):
        if order_tag not in self.dtensor_tile_groups:
            self.dtensor_tile_groups[order_tag] = DTensorTileGroup(order_tag)
        self.dtensor_tile_groups[order_tag].add_tensor_tile(tile, pe, interface_idx)

    def print(self):
        for order_tag, tiles in self.dtensor_tile_groups.items():
            print(f"<<<<< {order_tag} >>>>>")
            tiles.print()


class FIFO:
    def __init__(
        self,
        name: str,
        src: str,
        dst: list[str],
        data_shape: list[int],
        dtype: str,
        depth: int = 2,
    ):
        self.name = name
        self.src = src
        self.dst = sorted(dst)
        self.data_shape = data_shape
        self.dtype = dtype
        self.depth = depth

    def __str__(self):
        return f"FIFO({self.name}, src={self.src}, dst={self.dst}, {self.dtype}{self.data_shape}, depth={self.depth})"

    def __repr__(self):
        return self.__str__()


class FIFOManager:
    def __init__(self):
        self.fifos: list[FIFO] = []
        self.fifo_map: dict[tuple, FIFO] = {}

    def create_fifo(
        self, src: str, dst: list[str], data_shape: list[str], dtype: str
    ) -> FIFO:
        fifo = FIFO(
            name=f"fifo_{len(self.fifos)}",
            src=src,
            dst=dst,
            data_shape=data_shape,
            dtype=dtype,
        )
        self.fifos.append(fifo)
        return fifo

    def print(self):
        print("\n***** FIFOs *****")
        for key, fifo in self.fifo_map.items():
            print(f"{key}: {fifo}")
        print("***** FIFOs *****\n")


class SwitchNode:
    class Port:
        def __init__(
            self, id: int, data_shape: list[int], dtype: str, connected_nodes: list[str]
        ):
            self.id = id
            self.data_shape = data_shape
            self.dtype = dtype
            self.connected_nodes = connected_nodes
            self.bind_fifo: FIFO = None

        def bind_to_fifo(self, fifo: FIFO):
            assert (
                self.bind_fifo is None
            ), f"Port {self.id} already bound to {self.bind_fifo}"
            self.bind_fifo = fifo

        def __str__(self):
            return f"Port(data_shape={self.data_shape}, dtype={self.dtype}, connected_nodes={self.connected_nodes})"

        def __repr__(self):
            return self.__str__()

    class IntraConnect:
        def __init__(
            self, send_port_ids: list[int], recv_port_ids: list[int], offsets: list[int]
        ):
            self.send_port_ids = send_port_ids  # send_port_id
            self.recv_port_ids = recv_port_ids  # recv_port_id
            self.offsets = offsets

        def __str__(self):
            return f"(send:{self.send_port_ids} <=> recv:{self.recv_port_ids}, offsets={self.offsets})"

        def __repr__(self):
            return self.__str__()

    def __init__(self, name: str, send_port_num: int, recv_port_num: int):
        self.name = name
        self.max_send = send_port_num
        self.max_recv = recv_port_num
        self.send_ports: list[SwitchNode.Port] = []
        self.recv_ports: list[SwitchNode.Port] = []
        # connect send ports to recv ports
        self.intra_connect: list[SwitchNode.IntraConnect] = []

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name

    def print(self):
        print(f"\n<<<<< Switch {self.name} >>>>>")
        print(f"send ports: {self.send_ports}")
        print(f"recv ports: {self.recv_ports}")
        print(f"intra connect: {self.intra_connect}")


# ############################################################
# Computation Mapping Graph
# ############################################################
class OperationTagger:
    def __init__(self) -> None:
        self.tag_map: dict[str, str] = {}
        self.counter = 0

    def get_init_tag(self, key: str) -> str:
        """Return existing tag or assign a new one if not present."""
        if key not in self.tag_map:
            tag = f"tag_{self.counter}"
            self.tag_map[key] = tag
            self.counter += 1
        return self.tag_map[key]


class LiveDTensorTile:
    def __init__(self, tile: DTensorTile, token: str, is_input: bool):
        self.tile = tile
        self.token: str = (
            token  # LiveDTensorTiles with the same token should be processed in one 'run'
        )
        self.first_use = None
        self.last_use = None
        self.is_input: bool = is_input

    def __hash__(self):
        return hash((self.tile, self.first_use, self.last_use))

    def __eq__(self, other):
        return (
            self.tile == other.tile
            and self.first_use == other.first_use
            and self.last_use == other.last_use
        )

    def __str__(self):
        return (
            f"{self.token} {self.tile} [{self.first_use,self.last_use}] {self.is_input}"
        )

    def __repr__(self):
        return self.__str__()


class LiveDTensorTileGroup:
    """
    For each interface, classified by LiveDTensorTile token, follow the sequence of liveness range.
    """

    def __init__(self, live_dtensor_tiles: list[LiveDTensorTile]):
        self.is_input = live_dtensor_tiles[0].is_input
        self.dtensor_groups: dict[str, list[LiveDTensorTile]] = defaultdict(list)
        self.compatible_dtensor_ids: set[int] = set()
        for dtensor_tile in live_dtensor_tiles:
            self.dtensor_groups[dtensor_tile.token].append(dtensor_tile)
            self.compatible_dtensor_ids.add(dtensor_tile.tile.dtensor_id)
        for dtensor_groups in self.dtensor_groups.values():
            dtensor_groups.sort(key=lambda x: x.first_use)
            idx = 0
            while idx < len(dtensor_groups) - 1:
                assert (
                    dtensor_groups[idx].last_use <= dtensor_groups[idx + 1].first_use
                ), "liveness range overlapped."
                idx = idx + 1

    def grouping(self, tile_group: "LiveDTensorTileGroup") -> bool:
        # fixme: currently using tensor_id to check data type
        if self.is_input != tile_group.is_input:
            return False
        if self.compatible_dtensor_ids.isdisjoint(tile_group.compatible_dtensor_ids):
            return False
        if len(self.dtensor_groups) == len(tile_group.dtensor_groups):
            for token, dtensor_groups in self.dtensor_groups.items():
                if token not in tile_group.dtensor_groups:
                    return False
                merged = dtensor_groups + tile_group.dtensor_groups[token]
                merged.sort(key=lambda x: x.first_use)
                idx = 0
                while idx < len(dtensor_groups) - 1:
                    if dtensor_groups[idx].last_use > dtensor_groups[idx + 1].first_use:
                        return False
                    idx = idx + 1
        else:
            return False
        # group up
        for token, dtensor_groups in self.dtensor_groups.items():
            dtensor_groups.extend(tile_group.dtensor_groups[token])
        return True

    def __str__(self):
        ret = ""
        for token, tiles in self.dtensor_groups.items():
            ret += f"\t\t{token}: {tiles}"
        return ret

    def __repr__(self):
        return self.__str__()


# ------------------------------------------------------------
class NodeBase:
    node_list: list["NodeBase"] = []

    def __init__(
        self,
        name: str = None,
        func: func_d.FuncOp = None,
        tag: str = None,
        repeat: int = 0,
        length: int = 1,
    ):
        self.id = len(NodeBase.node_list)
        NodeBase.node_list.append(self)
        self.name = name if name is not None else f"function_{self.id}"
        self.func: func_d.FuncOp = func
        self.repeat: int = repeat
        self.length: int = length
        self.op_tag: str = tag
        # arg_idx -> tiling using arg as interface
        # TODO: interface reuse
        self.global_interfaces: dict[int, list[LiveDTensorTile]] = defaultdict(list)
        self.input_streams: list[Stream] = []
        self.output_streams: list[Stream] = []

    def is_isomorphic_to(self, other: "NodeBase") -> bool:
        # TODO: check in a more robust way
        if self is other:
            return True
        if self.op_tag != other.op_tag:
            return False
        in1 = Counter((s.src, s.type_str) for s in self.input_streams)
        in2 = Counter((s.src, s.type_str) for s in other.input_streams)
        if in1 != in2:
            return False
        out1 = Counter((s.src, s.type_str) for s in self.output_streams)
        out2 = Counter((s.src, s.type_str) for s in other.output_streams)
        if out1 != out2:
            return False
        return True

    def __str__(self) -> str:
        def fmt_list(lst: list) -> str:
            return "[" + ", ".join(str(item) for item in lst) + "]"

        return (
            f"Node({self.id}) {self.name}"
            f"Operation(tag='{self.op_tag}', repeat={self.repeat})\n"
            f"\tGlobal IO Tiles: "
            f"{ {k: fmt_list(v) for k, v in self.global_interfaces.items()} }\n"
            f"\tInput Streams: {[str(s) for s in self.input_streams]}\n"
            f"\tOutput Streams: {[str(s) for s in self.output_streams]}"
        )

    def __repr__(self) -> str:
        return self.__str__()


class InitialNode(NodeBase):
    def __init__(self, func: func_d.FuncOp, tag: str):
        super().__init__(func.attributes["sym_name"].value, func, tag, 1)

    def init_live_tile(self):
        """
        liveness analysis for global tiles used in the node
        # TODO: real liveness analysis
        """
        for live_tile_list in self.global_interfaces.values():
            for live_tile in live_tile_list:
                live_tile.first_use = 0
                live_tile.last_use = 9


class CollocatedNode(NodeBase):
    def __init__(
        self,
        tag: str,
        name: str = None,
        func: func_d.FuncOp = None,
        repeat: int = 0,
        length: int = 0,
    ):
        super().__init__(name=name, func=func, tag=tag, repeat=repeat, length=length)

    def _init_for_bundle(self, sample_node: NodeBase):
        self.input_streams = list(sample_node.input_streams)
        self.output_streams = list(sample_node.output_streams)
        self.global_interfaces = {key: [] for key in sample_node.global_interfaces}


# ------------------------------------------------------------
class ComputationGraph:
    def __init__(
        self,
        allo_module: allo_ir.ir.Module,
        stream_map: dict[str, Stream],
        core_func_args: dict[str, dict[int, tuple[Argument, bool]]],
    ):
        self.allo_module = allo_module
        self.nodes: dict[str, NodeBase] = {}
        self.edges: dict[str, Stream] = stream_map
        self.func_args = core_func_args

        self.tagger = OperationTagger()
        self.dependencies: dict[str, set[str]] = {}

        df_kernels = get_df_kernels(allo_module)
        # construct nodes
        for func in df_kernels:
            func_name = func.attributes["sym_name"].value
            tag_key = re.sub(
                r"func\.func\s+@[\w\d_]+(\s*\()", r"func.func\1", str(func.operation)
            )
            node = InitialNode(func, self.tagger.get_init_tag(tag_key))
            _, indexes = parse_kernel_name(func_name)
            params = core_func_args[func_name]
            for idx, (argument, is_input) in params.items():
                if argument.stream is not None:
                    if is_input:
                        node.input_streams.append(argument.stream)
                    else:
                        node.output_streams.append(argument.stream)
                if argument.dtensor is not None:
                    tensor_tile = DTensorTile(
                        argument.dtensor.global_id,
                        argument.dtensor.PE_tile_id_to_tensor_tile_id(indexes),
                    )
                    node.global_interfaces[idx].append(
                        LiveDTensorTile(tensor_tile, func_name, is_input)
                    )
            node.init_live_tile()
            self.nodes[func_name] = node
            self.dependencies[func_name] = set()
        # initiate dependencies
        for stream in self.edges.values():
            self.dependencies[stream.dst].add(stream.src)

    # ------------------------------------------------------------
    # Transformation Primitives
    # ------------------------------------------------------------

    def bundle(self, node_name_list: list[str]):
        """
        [A] [B] [C] [D]  => [A] x 4

        TODO: bundled nodes can be safely reordered
        """
        assert len(node_name_list) >= 2, "bundle at least two nodes"
        node_list: list[NodeBase] = []
        for name in node_name_list:
            assert name in self.nodes, f"Node({name}) not found"
            node_list.append(self.nodes[name])
        sample_node: NodeBase = node_list[0]
        for node in node_list:
            if not sample_node.is_isomorphic_to(node):
                raise ValueError(
                    f"Expect to bundle isomorphic nodes, Node({node.name}) is not isomorphic to Node({sample_node.name})"
                )
        bundled_node = CollocatedNode(
            tag=sample_node.op_tag,
            name=sample_node.name,
            func=sample_node.func,
            length=sample_node.length,
        )
        bundled_node._init_for_bundle(sample_node)
        for node in node_list:
            bundled_node.repeat += node.repeat
            for key, value in node.global_interfaces.items():
                assert key in bundled_node.global_interfaces
                bundled_node.global_interfaces[key].extend(value)

        # update stream
        for name, stream in self.edges.items():
            if stream.src in node_name_list:
                self.dependencies[stream.dst].pop(stream.src)
                stream.src = bundled_node.name
                self.dependencies[stream.dst].add(bundled_node.name)
            if stream.dst in node_name_list:
                stream.dst = bundled_node.name
                self.dependencies[bundled_node.name].add(stream.src)
        # update nodes and remove bundled function
        for name in node_name_list:
            removed = self.nodes.pop(name)
            if not name == bundled_node.name:
                removed.func.erase()
                self.func_args.pop(name)
                self.dependencies.pop(name)
        self.nodes[bundled_node.name] = bundled_node

    def chain(self, node_name_a: str, node_name_b: str):
        """
        [A] [B] => [[A]-[B]]
        """

        @dataclass
        class BufferizedStream:
            arg_idx_a: int
            arg_idx_b: int
            arg_a: Value
            arg_b: Value
            stream_input: list[Value]
            stream_output: list[Value]

        node_a, node_b = self.nodes.pop(node_name_a), self.nodes.pop(node_name_b)
        param_a, param_b = self.func_args[node_name_a], self.func_args[node_name_b]
        assert node_a is not None and node_b is not None, "node not found"
        if node_name_b in self.dependencies[node_name_a]:
            raise ValueError(
                f"Cannot chain Node({node_name_a}) and Node({node_name_b})"
            )
        # TODO: repeat function
        assert (
            node_a.repeat == node_b.repeat == 1
        ), "Cannot chaining nodes with repeats currently"
        bundled_tag = (
            f"({node_a.op_tag})x{node_a.repeat}-({node_b.op_tag})x{node_b.repeat}"
        )
        chained_node = CollocatedNode(
            bundled_tag,
            name=f"{node_a.name}-{node_b.name}",
            repeat=1,
            length=node_a.length + node_b.length,
        )
        bufferized_stream: dict[Stream, BufferizedStream] = {}
        node_a.output_streams = [
            stream for stream in node_a.output_streams if stream.dst != node_name_b
        ]
        kept_streams = []
        for stream in node_b.input_streams:
            if stream.src == node_name_a:
                idx_a, idx_b = -1, -1
                for idx, arg_info in param_a.items():
                    if arg_info[0].stream is not None and arg_info[0].stream == stream:
                        idx_a = idx
                        break
                for idx, arg_info in param_b.items():
                    if arg_info[0].stream is not None and arg_info[0].stream == stream:
                        idx_b = idx
                        break
                assert idx_a >= 0 and idx_b >= 0
                bufferized_stream[stream] = BufferizedStream(
                    idx_a, idx_b, None, None, [], []
                )
            else:
                kept_streams.append(stream)
        node_b.input_streams = kept_streams
        chained_node.input_streams.extend(node_a.input_streams)
        chained_node.input_streams.extend(node_b.input_streams)
        chained_node.output_streams.extend(node_a.output_streams)
        chained_node.output_streams.extend(node_b.output_streams)
        # refactor function
        function_a: func_d.FuncOp = node_a.func
        function_b: func_d.FuncOp = node_b.func
        # - function parameters
        in_types_a: list = function_a.attributes["function_type"].value.inputs
        arg_idx_offset = len(in_types_a)
        in_types_b: list = function_b.attributes["function_type"].value.inputs
        out_types_a = function_a.attributes["function_type"].value.results
        out_types_b = function_b.attributes["function_type"].value.results
        chained_node.global_interfaces.update(node_a.global_interfaces)
        for key, value in node_b.global_interfaces.items():
            for live_tile in value:
                live_tile.first_use += Config.LOCAL_CODE_OFFSET * node_a.length
                live_tile.last_use += Config.LOCAL_CODE_OFFSET * node_a.length
            chained_node.global_interfaces[arg_idx_offset + key] = value
        new_token = node_a.name + "-" + node_b.name
        for live_tile_list in chained_node.global_interfaces.values():
            for live_tile in live_tile_list:
                live_tile.token = new_token
        with function_a.context, allo_ir.ir.Location.unknown():
            func_type = FunctionType.get(
                in_types_a + in_types_b, out_types_a + out_types_b
            )
            new_function = func_d.FuncOp(
                chained_node.name,
                func_type,
                ip=InsertionPoint(function_a),
            )
            new_function.attributes["df.kernel"] = UnitAttr.get()
            entry_block = new_function.add_entry_block()
            for old, new in zip(
                function_a.arguments + function_b.arguments, new_function.arguments
            ):
                old.replace_all_uses_with(new)
            for bufferized_stream_info in bufferized_stream.values():
                bufferized_stream_info.arg_a = new_function.arguments[
                    bufferized_stream_info.arg_idx_a
                ]
                bufferized_stream_info.arg_b = new_function.arguments[
                    bufferized_stream_info.arg_idx_b + arg_idx_offset
                ]
            with InsertionPoint(entry_block):
                for func_block in function_a.body:
                    for op in func_block.operations:
                        if isinstance(op, func_d.ReturnOp):
                            assert len(op.operands_) == 0
                            continue
                        new_op = op.clone()
                        for old, new in zip(op.results, new_op.results):
                            old.replace_all_uses_with(new)
                for func_block in function_b.body:
                    for op in func_block.operations:
                        new_op = op.clone()
                        for old, new in zip(op.results, new_op.results):
                            old.replace_all_uses_with(new)
                function_a.erase()
                function_b.erase()
            # bufferize streams
            for stream, bufferized_stream_info in bufferized_stream.items():
                stream_puts = [
                    use.owner
                    for use in bufferized_stream_info.arg_a.uses
                    if isinstance(use.owner, allo_d.StreamPutOp)
                ]
                stream_gets = [
                    use.owner
                    for use in bufferized_stream_info.arg_b.uses
                    if isinstance(use.owner, allo_d.StreamGetOp)
                ]
                assert len(stream_puts) == len(stream_gets)
                for i in range(len(stream_puts)):
                    stream_put: allo_d.StreamPutOp = stream_puts[i]
                    stream_get: allo_d.StreamGetOp = stream_gets[i]
                    # TODO: support bufferize stream in branches or even loops
                    assert isinstance(
                        stream_put.parent.opview, func_d.FuncOp
                    ) and isinstance(
                        stream_get.parent.opview, func_d.FuncOp
                    ), "Only support bufferize stream in the main body"
                    put_value = stream_put.operands[-1]
                    get_result = stream_get.result
                    get_result.replace_all_uses_with(put_value)
                    stream_put.erase()
                    stream_get.erase()
                # update argument info
                param_a.pop(bufferized_stream_info.arg_idx_a)
                param_b.pop(bufferized_stream_info.arg_idx_b)
                self.edges.pop(stream.name)
        chained_node.func = new_function
        self.func_args.pop(node_name_a)
        self.func_args.pop(node_name_b)
        self.func_args[chained_node.name] = param_a
        for key, value in param_b.items():
            self.func_args[chained_node.name][arg_idx_offset + key] = value

        dep = self.dependencies.pop(node_name_a)
        dep.update(self.dependencies.pop(node_name_b))
        self.dependencies[chained_node.name] = dep
        for deps in self.dependencies.values():
            if node_name_a in deps:
                deps.remove(node_name_a)
                deps.add(chained_node.name)
            if node_name_b in deps:
                deps.remove(node_name_b)
                deps.add(chained_node.name)
        for stream in self.edges.values():
            if stream.src == node_name_a or stream.src == node_name_b:
                stream.src = chained_node.name
            if stream.dst == node_name_a or stream.dst == node_name_b:
                stream.dst = chained_node.name
        self.nodes[chained_node.name] = chained_node

    # ------------------------------------------------------------
    # Graph Information
    # ------------------------------------------------------------
    def get_global_io(
        self,
    ) -> tuple[dict[str, dict[int, LiveDTensorTileGroup]], dict[str, dict[int, int]]]:
        global_tile_io: dict[str, dict[int, LiveDTensorTileGroup]] = {}
        arg_idx_to_interface: dict[str, dict[int, int]] = {}
        for name, node in self.nodes.items():
            input_interface, output_interface = 0, 0
            dict_: dict[int, LiveDTensorTileGroup] = {}
            idx_to_interface: dict[int, int] = {}
            for idx, interfaces in node.global_interfaces.items():
                dict_[idx] = LiveDTensorTileGroup(interfaces)
                idx_to_interface[idx] = idx
                if interfaces[0].is_input:
                    input_interface += 1
                else:
                    output_interface += 1
            # try to satisfy compute tile port resource constraint
            while (
                input_interface > Config.COMPUTE_MAX_RECV
                or output_interface > Config.COMPUTE_MAX_SEND
            ):
                changed = False
                keys = list(dict_.keys())
                for i in range(len(keys)):
                    for j in range(i + 1, len(keys)):
                        key_1, key_2 = keys[i], keys[j]
                        value_1, value_2 = dict_[key_1], dict_[key_2]
                        if value_1.grouping(value_2):
                            changed = True
                            if value_1.is_input:
                                input_interface -= 1
                            else:
                                output_interface -= 1
                            del dict_[key_2]
                            for idx in idx_to_interface.keys():
                                if idx_to_interface[idx] == key_2:
                                    idx_to_interface[idx] = key_1
                            break
                    if changed:
                        break
                if not changed:
                    raise ValueError(
                        f"Invalide compute kernel {name}, port number exceeded."
                    )

            global_tile_io[name] = dict_
            arg_idx_to_interface[name] = idx_to_interface
        return global_tile_io, arg_idx_to_interface

    def get_node_dependencies(self) -> dict[str, set[str]]:
        dependencies: dict[str, set[str]] = {key: set() for key in self.nodes.keys()}
        for stream in self.edges.values():
            dependencies[stream.dst].add(stream.src)
        return dependencies

    def get_connections(self) -> dict[tuple[str, str], int]:
        connections: dict[tuple[str, str], int] = {}
        for stream in self.edges.values():
            id_1, id_2 = self.nodes[stream.src].id, self.nodes[stream.dst].id
            if id_1 > id_2:
                key = (stream.dst, stream.src)
            else:
                key = (stream.src, stream.dst)
            if key in connections:
                connections[key] += 1
            else:
                connections[key] = 1
        connection_info: list[tuple[int, str, str]] = []
        for (name_1, name_2), count in connections.items():
            connection_info.append((count, name_1, name_2))
        return connection_info
