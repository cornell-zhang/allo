import numpy as np
from typing import Dict, List, Tuple, Set
from collections import defaultdict

from ..utils import format_str
from ..._mlir.dialects import func as allo_func_d
from ...memory import DTensor

from .utils import get_element_type
from ..aie import map_kernels_to_device_mesh

# =======================
import aie.dialects.aie as aie_d
import aie.dialects.aiex as aiex_d
import aie.dialects.aievec as aievec_d
import aie.dialects.arith as aie_arith_d
import aie.dialects.func as aie_func_d
import aie.dialects.scf as aie_scf_d

import aie.ir as aie_ir
# =======================
from dataclasses import dataclass

@dataclass(frozen=True)
class DMATensorTile:
    dtensot_tile_id: int # dTensor may need to be further partitioned
    shim_id: int
    mem_id: int
    tensor_tile_labels: List
    offset: List
    size: List
    stride: List

def map_global_io(inputs, outputs) -> Tuple[Dict[str,List[DMATensorTile]] , int, int]:
    """
    Current constrians:
        - use 4 mem-shim tile pairs for io
        - each port is assigned to one dtensor tile

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
    
    used_tiles:List[Tile] = []

    def assign_tile(send_need, recv_need) -> int: 
        """
            Try to assign a memory tile satisfying the requirement.
            Return the tile index.
                -1 indicates no tile avaliable.
        """
        # 1. Attempt to use a new memory tile
        if (len(used_tiles) < MAX_MEM_TILES and send_need <= MAX_SEND and recv_need <= MAX_RECV):
            used_tiles.append(Tile(send_need, recv_need))
            return len(used_tiles) - 1  
        # 2. Otherwise, try to pack into an existing tile
        for i, _ in enumerate(used_tiles):
            if (used_tiles[i].send_number + send_need <= MAX_SEND and used_tiles[i].recv_number + recv_need <= MAX_RECV):
                used_tiles[i].send_number += send_need
                used_tiles[i].recv_number += recv_need
                return i 
        # 3. No tile fits
        return -1

    def map_dtensor_to_tile(dtensor:DTensor, is_input:bool):
        """
            Currently, we focus on dtensor io using memory tiles. 
            Shim tiles are assigned using a one-to-one mapping from memory tiles.

            DTensors are sent to or from compute cores. 
            Since memory tile is used for transfer, we assume that `receive` implies one `send` and `send` implies one `receive`.
        """
        device_dims, size, stride = dtensor.get_access_pattern()
        tensor_tiles = list(dtensor.global_placement.keys()) # 'R' can use one port yet multilple destinations

        send_need = len(tensor_tiles) if is_input else 1
        recv_need = 1 if is_input else len(tensor_tiles)
        mem_tile_id = assign_tile(send_need, recv_need)
        if mem_tile_id >= 0:
            return [DMATensorTile(
                0, mem_tile_id, mem_tile_id, 
                tensor_tiles, [0, 0, 0, 0], size, stride)]
        # We failed to transfer the whole tensor with one memory tile. Try using more.
        dma_tensor_tiles:List[DMATensorTile] = []
        # fixme: incomplete
        #   Currently, we may allow tensor tiles on a sharding demension to be sent using different memory tiles
        lose_factor = 1 if len(device_dims) <= 1 else size[device_dims[0]] 
        remaining = tensor_tiles[:]
        start_idx = 0
        while remaining:
            offset = [0,0,0,0]
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
            if len(device_dims) == 1:
                offset[device_dims[0]] = start_idx
                size[device_dims[0]] = len(chunk)
            else:
                offset[device_dims[1]] = start_idx // lose_factor
                size[device_dims[1]] = len(chunk) // lose_factor
            dma_tensor_tiles.append(
                DMATensorTile(len(dma_tensor_tiles), mem_tile_id, mem_tile_id, chunk, offset, size, stride)
            )
            remaining = remaining[len(chunk) :]
            start_idx += len(chunk)
        return dma_tensor_tiles

    tile_map:Dict[str,List[DMATensorTile]] = defaultdict(list)

    for io_lst, is_input in ((inputs, True), (outputs, False)):
        for _, sub in io_lst.items():
            for dtensor in sub["_global"]:
                tile_map[dtensor.name].extend(map_dtensor_to_tile(dtensor, is_input=is_input))

    return tile_map, len(used_tiles), len(used_tiles)

class CodeGenerator:
    def __init__(
        self, device_type:str,
    ):
        self.device_type = device_type
        self.tile_map:Dict[str,aie_d.TileOp] = {}
        self.fifo_map:Dict[str,aie_d.object_fifo] = {}
        # function name (with id) -> a map from DTensor to fifo name
        self.compute_core_io:Dict[str:Dict[DTensor,str]] = {}
        self.external_functions:str = ""

        self.aie_module = None
        self.global_ip: aie_ir.InsertionPoint = None # mark the inserting point for buffers
    
    def preporocess_dumped_core_func(self, original_func: allo_func_d.FuncOp) -> str:
        # declare external kernel function before use
        func_str = self.external_functions + "\n" + str(original_func)
        # TODO: resolve `allo` (unregistered dialect)
        return func_str
    
    def build_core_function(
        self,
        func_core: aie_d.Core,
        original_func: allo_func_d.FuncOp,
        func_args:List[Tuple[DTensor,bool]]
    ):
        original_module = aie_ir.Module.parse(self.preporocess_dumped_core_func(original_func))
        parsed_function:aie_func_d.FuncOp = None
        for func in original_module.body.operations:
            if isinstance(func, aie_func_d.FuncOp):
                if not ("sym_visibility" in func.attributes
                    and func.attributes["sym_visibility"].value == "private"
                ):
                    if parsed_function is None:
                        parsed_function = func
                    else:
                        raise ValueError("Too many core functions. Fail to resolve.")
        assert not parsed_function is None
        print("parsed_function\n", parsed_function)
        
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
                io_map =self.compute_core_io[parsed_function.name.value]
                for i in range(len(parsed_function.arguments)):
                    arg_info:Tuple[DTensor,bool] = func_args[i]
                    # [NOTE]: modified from object_fifo.acquire
                    subview_t = aie_d.ObjectFifoSubviewType.get(self.fifo_map[io_map[arg_info[0]]].datatype)
                    acq = aie_d.ObjectFifoAcquireOp(
                        subview_t, 1 if arg_info[1] else 0, self.fifo_map[io_map[arg_info[0]]].sym_name.value, 1)
                    acquired = aie_d.ObjectFifoSubviewAccessOp(
                        parsed_function.arguments[i].type, acq.subview, acq.size.value - 1
                    ).result
                    parsed_function.arguments[i].replace_all_uses_with(acquired)
                 
                # parsed_function.arguments[0].replace_all_uses_with(parsed_function.arguments[1])
                for parsed_func_block in parsed_function.body:
                    for op in parsed_func_block.operations:
                        if op.name == "func.return":
                            continue
                        if op.name == "memref.alloc":
                            buffer_op = aie_d.BufferOp(
                                buffer=op.results[0].type, 
                                tile=self.tile_map[f"compute_{parsed_function.name.value}"], 
                                ip = self.global_ip
                            )
                            for old, new in zip(op.results, buffer_op.results):
                                old.replace_all_uses_with(new)
                            continue
                        new_op = op.clone()
                        for old, new in zip(op.results, new_op.results):
                            old.replace_all_uses_with(new)
                
                # release
                for i in range(len(parsed_function.arguments)):
                    arg_info:Tuple[DTensor,bool] = func_args[i]
                    self.fifo_map[io_map[arg_info[0]]].release(1 if arg_info[1] else 0,1)
                
                aie_scf_d.YieldOp([])
            aie_d.EndOp()
                
    def aie_codegen(
        self,
        core_func_groups:Dict[str, List[allo_func_d.FuncOp]],
        external_funcs:List[allo_func_d.FuncOp],
        inputs, outputs,
        use_external_kernels:Dict[str,bool],
        stream_info:Dict,
        core_func_args:Dict[str,Tuple[DTensor,bool]]
    )-> aie_ir.Module:
        
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
                    self.tile_map[f"shim_{shim_id}"] = aie_d.TileOp(
                        col=shim_id, row=0,
                        allocation_scheme="shim",
                    )
                # mem tiles
                for mem_id in range(mem_tile_num):
                    self.tile_map[f"mem_{mem_id}"] = aie_d.TileOp(
                        col=mem_id, row=1,
                        allocation_scheme="mem", 
                    )
                # compute tiles
                # 'logic' mapping for different function groups.
                mappings = {}
                for func_name in core_func_groups:
                    if len(inputs[func_name]["_global"]) > 0:
                        mappings[func_name] = inputs[func_name]["_global"][0].mapping
                    else:
                        mappings[func_name] = outputs[func_name]["_global"][0].mapping
                aie_mesh = (5, 4)
                for func_name, tile_ids in map_kernels_to_device_mesh(mappings, aie_mesh).items():
                    for idx, func in zip(tile_ids, core_func_groups[func_name]):
                        self.tile_map[f"compute_{func.attributes["sym_name"].value}"] = aie_d.TileOp(
                            col=idx[0], row=idx[1]+2,
                        )

                print(self.tile_map)
                for io, arg_lst in (("in", inputs), ("out", outputs)):
                    for func_name, sub_func_lst in arg_lst.items():
                        for idx, dtensor in enumerate(sub_func_lst["_global"]):
                            placement = dtensor.global_placement
                            # shim <-> mem (one to one)
                            for dma_tile in io_mapping[dtensor.name]:
                                print(dma_tile)
                                # define objectfifo
                                name = f"{io}_shim_{dtensor.name}{dma_tile.dtensot_tile_id}"
                                producer = self.tile_map[f"shim_{dma_tile.shim_id}"] if io == 'in' else self.tile_map[f"mem_{dma_tile.mem_id}"]
                                consumer = [self.tile_map[f"mem_{dma_tile.mem_id}"]] if io == 'in' else [self.tile_map[f"shim_{dma_tile.shim_id}"]]
                                memref_type = aie_ir.MemRefType.get(dma_tile.size, get_element_type(str(dtensor.dtype)))
                                fifo_shim = self.fifo_map[name] = aie_d.object_fifo(name, producer, consumer, depth = 2, datatype = memref_type)

                                # mem <-> compute (one to ?)
                                fifo_mem = []
                                mem_stride = [0]
                                mem_tile = self.tile_map[f"mem_{dma_tile.mem_id}"]
                                for tensor_tile in dma_tile.tensor_tile_labels:
                                    print("placement[tensor_tile]", tensor_tile, placement[tensor_tile])
                                    # distribute to placement[tensor_tile]
                                    compute_tiles = []
                                    name = f"{io}_mem_{dtensor.name}_{tensor_tile}"
                                    for tile in placement[tensor_tile]:
                                        idx_str = "_".join(map(str, tile))
                                        # seems confusing. "sym_name" is parsed in this way
                                        compute_tiles.append(self.tile_map[f"compute_{func_name}_{idx_str}"])
                                        self.compute_core_io.setdefault(f'{func_name}_{idx_str}', {})[dtensor]=name
                                    print("compute_cores", compute_tiles)
                                    if io == 'in':
                                        producer = mem_tile
                                    else:
                                        # fixme: only one valid producer
                                        # assert len(compute_tiles)==1
                                        producer = compute_tiles[0]
                                    consumer = compute_tiles if io == 'in' else [mem_tile]
                                    fifo = self.fifo_map[name] = aie_d.object_fifo(name, producer, consumer, depth = 2, datatype = memref_type)
                                    fifo_mem.append(fifo)
                                    mem_stride.append(
                                        mem_stride[-1] + np.prod(dtensor.get_local_shape())
                                    )
                                aie_d.object_fifo_link(
                                    fifo_shim if io == "in" else fifo_mem,
                                    fifo_mem if io == "in" else fifo_shim,
                                    [] if io == "in" else mem_stride[:-1],
                                    mem_stride[:-1] if io == "in" else []
                                )
                # compute <-> compute
                # TODO: allo stream
                
                for func_name in core_func_groups:
                    for func in core_func_groups[func_name]:
                        func_name_w_id = func.attributes["sym_name"].value
                        func_core = aie_d.Core(
                            tile = self.tile_map[f"compute_{func_name_w_id}"],
                            link_with = "external.o" if use_external_kernels[func_name_w_id] else None
                        )
                        if self.global_ip == None:
                            self.global_ip = aie_ir.InsertionPoint(func_core)
                        self.build_core_function(func_core,func,core_func_args[func_name_w_id])
        print("\n")
        print(self.aie_module)
        return self.aie_module
