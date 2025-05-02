import os
import re
from typing import Dict, List, Tuple, Set
from collections import defaultdict

from .utils import format_str, format_code, print_module
import allo._mlir._mlir_libs._mlir as allo_ir
from .._mlir.dialects import (
    allo as allo_d,
    func as func_d,
)

from .._mlir.ir import (
    MemRefType,
    InsertionPoint,
    FlatSymbolRefAttr,
    StringAttr,
)
from .._mlir.passmanager import PassManager as mlir_pass_manager

from ..passes import analyze_read_write_patterns
from .aie import map_kernels_to_device_mesh
from ..memory import Layout, DTensor
from dataclasses import dataclass

# =======================
import aie.dialects.aie as aie_d
import aie.dialects.aiex as aiex_d
import aie.dialects.aievec as aievec_d
import aie.dialects.arith as aie_arith_d
import aie.dialects.func as aie_func_d
import aie.dialects.scf as aie_scf_d

import aie.ir as aie_ir
# =======================

# ------------------------------------------------------------------
@dataclass(frozen=True)
class DMATensorTile:
    dtensot_tile_id: int # dTensor may need to be further partitioned
    shim_id: int
    mem_id: int
    tensor_tile_labels: List
    offset: List
    size: List
    stride: List

# ------------------------------------------------------------------

def inject_external_kernels(module: allo_ir.ir.Module) -> Dict[str,bool]:
    use_external_kernels = {}
    injected_kernels = set()
    with module.context, allo_ir.ir.Location.unknown():
        for func in module.body.operations:
            if isinstance(func, func_d.FuncOp) and (
                "sym_visibility" not in func.attributes
                or func.attributes["sym_visibility"].value != "private"
            ):
                if func.attributes["sym_name"].value != "top":
                    func_name: str = func.attributes["sym_name"].value
                    use_external_kernels[func_name] = False
                    for block in func.regions[0].blocks:
                        for op in block.operations:
                            # vec add/mul
                            if op.operation.name in {"linalg.add", "linalg.mul"}:
                                op_name = op.operation.name.split(".")[1]
                                dtype = str(op.inputs[0].type.element_type)
                                shape = MemRefType(op.inputs[0].type).shape
                                kernel_name = f"{op_name}_{dtype}_vector"
                                use_external_kernels[func_name] = True
                            # matmul
                            elif op.operation.name == "linalg.matmul":
                                M, K = MemRefType(op.inputs[0].type).shape
                                _, N = MemRefType(op.inputs[1].type).shape
                                dtype = str(op.inputs[0].type.element_type)
                                out_dtype = str(op.outputs[0].type.element_type)
                                if (dtype, out_dtype) not in [
                                    ("i8", "i8"), ("i16", "i16"), ("i16", "i32"),
                                    ("bf16", "bf16"), ("bf16", "f32"),
                                ]:
                                    continue
                                kernel_name = f"matmul_scalar_{dtype}_{out_dtype}"
                                use_external_kernels[func_name] = True
                            else:
                                continue  
                            
                            # Inject AIE kernel
                            func_type = func_d.FunctionType.get(
                                [op.inputs[0].type, op.inputs[1].type, op.outputs[0].type], [],
                            )
                            # replace operation
                            func_d.CallOp(
                                [], FlatSymbolRefAttr.get(kernel_name),
                                [op.inputs[0], op.inputs[1], op.outputs[0]],
                                ip=InsertionPoint(op),
                            )
                            op.erase()
                            # register external kernel
                            if kernel_name in injected_kernels:
                                continue
                            injected_kernels.add(kernel_name)
                            kernel = func_d.FuncOp(
                                kernel_name, func_type, ip=InsertionPoint(func),
                            )
                            kernel.attributes["sym_visibility"] = StringAttr.get("private")

    return use_external_kernels

def classify_aie_functions(
    module: allo_ir.ir.Module
) -> Tuple[func_d.FuncOp, Dict[str, List[func_d.FuncOp]], List[func_d.FuncOp]]:
    # top function
    top_func:func_d.FuncOp = None
    # core functions
    core_func_groups:Dict[str, List[func_d.FuncOp]] = {}
    # external functions
    external_funcs:List[func_d.FuncOp] = []
    with module.context, allo_ir.ir.Location.unknown():
        for func in module.body.operations:
            if isinstance(func, func_d.FuncOp):
                if ("sym_visibility" in func.attributes
                    and func.attributes["sym_visibility"].value == "private"
                ):
                    external_funcs.append(func)
                else:
                    if func.attributes["sym_name"].value == "top":
                        top_func = func
                    else:
                        func_name_w_id = func.attributes["sym_name"].value
                        func_name = re.match(r"^(.*?)_\d", func_name_w_id).group(1)
                        if func_name not in core_func_groups:
                            core_func_groups[func_name] = []
                        core_func_groups[func_name].append(func)
    return top_func, core_func_groups, external_funcs

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
        placement, device_dims, size, stride = dtensor.get_access_pattern()
        tensor_tiles = list(placement.keys()) # 'R' can use one port yet multilple destinations

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

def get_element_type(dtype_str: str):
        if dtype_str == "i32":
            return aie_ir.IntegerType.get_signless(32)
        elif dtype_str == "i16":
            return aie_ir.IntegerType.get_signless(16)
        elif dtype_str == "i8":
            return aie_ir.IntegerType.get_signless(8)
        elif dtype_str == "f32":
            return aie_ir.F32Type.get()
        elif dtype_str == "f16":
            return aie_ir.F16Type.get()
        else:
            raise ValueError(f"Unsupported dtype: {dtype_str}")
        
class CodeGenerator:
    def __init__(
        self, device_type:str,
    ):
        self.device_type = device_type
        self.tile_map:Dict[str,aie_d.TileOp] = {}
        self.fifo_map:Dict[str,aie_d.object_fifo] = {}
        self.external_functions:str = ""
        self.aie_module = None
        self.global_ip: aie_ir.InsertionPoint = None # mark the inserting point for buffers
    
    def preporocess_dumped_core_func(self, original_func: func_d.FuncOp) -> str:
        # declare external kernel function before use
        func_str = self.external_functions + "\n" + str(original_func)
        # TODO: resolve `allo` (unregistered dialect)
        func_str = '''
            func.func @core_0(%arg0: memref<1024xi32>, %arg1: memref<1024xi32>) attributes {df.kernel, itypes = "ss", otypes = "", stypes = "__"} {
                %c1_i32 = arith.constant 1 : i32
                %alloc = memref.alloc() : memref<i32>
                memref.store %c1_i32, %alloc[] : memref<i32>
                %alloc_0 = memref.alloc() : memref<1024xi32>
                %c0 = arith.constant 0 : index
                %c1024 = arith.constant 1024 : index
                %c1 = arith.constant 1 : index
                scf.for %arg2 = %c0 to %c1024 step %c1 {
                %0 = memref.load %alloc[] : memref<i32>
                memref.store %0, %alloc_0[%arg2] : memref<1024xi32>
                }
                %alloc_1 = memref.alloc() : memref<1024xi32>
                memref.copy %alloc_1, %alloc_0 {to = "B"} : memref<1024xi32> to memref<1024xi32>
                return
            }
        '''
        return func_str
    
    def build_core_function(
        self,
        func_core: aie_d.Core,
        original_func: func_d.FuncOp
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
            # insert operations to get 'function parameter'

            # TODO: replace alloc with buffer
            for parsed_func_block in parsed_function.body:
                with aie_ir.InsertionPoint(loop.body):
                    for op in parsed_func_block.operations:
                        if op.name == "func.return":
                            continue
                        new_op = op.clone()
                        for old, new in zip(op.results, new_op.results):
                            old.replace_all_uses_with(new)
            print(self.aie_module)
            
    def aie_codegen(
        self,
        core_func_groups:Dict[str, List[func_d.FuncOp]],
        external_funcs:List[func_d.FuncOp],
        inputs, outputs,
        use_external_kernels:Dict[str,bool]
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
                mappings = {}
                for func_name in core_func_groups:
                    if len(inputs[func_name]["_global"]) > 0:
                        mappings[func_name] = inputs[func_name]["_global"][0].mapping
                    else:
                        mappings[func_name] = outputs[func_name]["_global"][0].mapping
                aie_mesh = (5, 4)
                print("mapping",mappings)
                for func_name, tile_ids in map_kernels_to_device_mesh(mappings, aie_mesh).items():
                    for idx, func in zip(tile_ids, core_func_groups[func_name]):
                        self.tile_map[f"compute_{func.attributes["sym_name"].value}"] = aie_d.TileOp(
                            col=idx[0], row=idx[1]+2,
                        )

                print(self.tile_map)
                # TODO: fifo
                for io, arg_lst in (("in", inputs), ("out", outputs)):
                    for func_name, sub_func_lst in arg_lst.items():
                        for idx, dtensor in enumerate(sub_func_lst["_global"]):
                            mapping = dtensor.mapping
                            placement = dtensor.layout.get_placement(mapping)
                            # shim <-> mem (one to one)
                            for dma_tile in io_mapping[dtensor.name]:
                                print(dma_tile)
                                # define objectfifo
                                name = f"{io}_shim_{dtensor.name}{dma_tile.dtensot_tile_id}"
                                producer = self.tile_map[f"shim_{dma_tile.shim_id}"] if io == 'in' else self.tile_map[f"mem_{dma_tile.mem_id}"]
                                consumer = [self.tile_map[f"mem_{dma_tile.mem_id}"]] if io == 'in' else [self.tile_map[f"shim_{dma_tile.shim_id}"]]
                                memref_type = aie_ir.MemRefType.get(dma_tile.size, get_element_type(str(dtensor.dtype)))
                                self.fifo_map[name]  = aie_d.object_fifo(name, producer, consumer, depth = 2, datatype = memref_type)
                                # mem <-> compute (one to ?)
                                print(mapping)
                                for tensor_tile in dma_tile.tensor_tile_labels:
                                    print("placement[tensor_tile]", tensor_tile, placement[tensor_tile])
                                    # diatribute to placement[tensor_tile]
                                    for tile in placement[tensor_tile]:
                                        for dt in sub_func_lst[tile]:
                                            print(dt)
                                        print()
                                    

                # shim to mem tile
                # TODO: buffer
                
                for func_name in core_func_groups:
                    for func in core_func_groups[func_name]:
                        func_name_w_id = func.attributes["sym_name"].value
                        func_core = aie_d.Core(
                            tile = self.tile_map[f"compute_{func_name_w_id}"],
                            link_with = "external.o" if use_external_kernels[func_name_w_id] else None
                        )
                        if self.global_ip == None:
                            self.global_ip = aie_ir.InsertionPoint(func_core)
                        self.build_core_function(func_core,func)
                    
        return self.aie_module

class AIE_MLIRModule:
    def __init__(
        self,
        module: allo_ir.ir.Module,
        top_func_name:str,
        func_args:dict,
        project_dir:str,
        stream_info:dict
    ):
        self.allo_module:allo_ir.ir.Module = module
        self.top_func_name:str = top_func_name
        self.func_args:dict = func_args
        self.stream_info:dict = stream_info
        self.project_dir:str = project_dir

        self.global_inputs:Set = set()
        self.global_outputs:Set = set()

        self.aie_module:aie_ir.Module = None
        print(module)
    
    def collect_io(
        self,
        func_groups:Dict[str, List[func_d.FuncOp]],
    )-> Tuple[Dict,Dict]:
        inputs = {}
        outputs = {}
        for func_name, funcs in func_groups.items():
            inputs[func_name] = {}
            outputs[func_name] = {}
            inputs[func_name]["_global"] = []
            outputs[func_name]["_global"] = []
            for func in funcs:
                func_name_w_id = func.attributes["sym_name"].value
                func_id = tuple(map(int, func_name_w_id.split(func_name + "_")[-1].split("_")))
                # fixme: `analyze_read_write_patterns` considers parameters that are both read and written as outputs
                in_idx, out_idx = analyze_read_write_patterns(func)
                for io_lst, io_idx, io in ((inputs, in_idx, "in"), (outputs, out_idx, "out")):
                    io_lst[func_name][func_id] = []
                    for idx in io_idx:
                        dtensor = self.func_args[func_name_w_id][idx]
                        if dtensor not in io_lst[func_name]["_global"]:
                            io_lst[func_name]["_global"].append(dtensor)
                            if io == "in":
                                self.global_inputs.add(dtensor)
                            else:
                                self.global_outputs.add(dtensor) 
                        io_lst[func_name][func_id].append(dtensor)
        return inputs, outputs
        
    def build(self, device_type = "npu1_4col"):
        os.makedirs(os.path.join(self.project_dir, "build"), exist_ok=True)
        # record original allo mlir
        with open(
            os.path.join(self.project_dir, "original.mlir"), "w", encoding="utf-8"
        ) as f:
            f.write(str(self.allo_module))
        # - extract external kernels
        use_external_kernels = inject_external_kernels(self.allo_module)
        # - lower tensor to memref with registered pass
        passes = ["func.func(convert-linalg-to-affine-loops),lower-affine",]
        pipeline = f'builtin.module({",".join(passes)})'
        with self.allo_module.context:
            mlir_pass_manager.parse(pipeline).run(self.allo_module.operation)
        print_module(self.allo_module)
        print()
        print(self.allo_module)
        top_func, core_func_groups, external_funcs = classify_aie_functions(self.allo_module)
        inputs, outputs = self.collect_io(core_func_groups)

        code_generator = CodeGenerator(device_type)
        self.aie_module = code_generator.aie_codegen(
            core_func_groups, external_funcs,
            inputs, outputs,
            use_external_kernels
        )
        # TODO

    def __call__(self, *args):
        pass
        # TODO