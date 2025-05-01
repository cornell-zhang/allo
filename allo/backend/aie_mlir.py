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
from ..memory import Layout, DTensor
from dataclasses import dataclass

# =======================
import aie.dialects.aie as aie_d
import aie.dialects.aiex as aiex_d
import aie.dialects.aievec as aievec_d
import aie.dialects.func as aie_func_d

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

def inject_external_kernels(module: allo_ir.ir.Module) -> Dict:
    external_kernels = {}
    injected_kernels = set()
    with module.context, allo_ir.ir.Location.unknown():
        for func in module.body.operations:
            if isinstance(func, func_d.FuncOp) and (
                "sym_visibility" not in func.attributes
                or func.attributes["sym_visibility"].value != "private"
            ):
                if func.attributes["sym_name"].value != "top":
                    func_name: str = func.attributes["sym_name"].value
                    external_kernels[func_name] = []
                    for block in func.regions[0].blocks:
                        for op in block.operations:
                            # vec add/mul
                            if op.operation.name in {"linalg.add", "linalg.mul"}:
                                op_name = op.operation.name.split(".")[1]
                                dtype = str(op.inputs[0].type.element_type)
                                shape = MemRefType(op.inputs[0].type).shape
                                kernel_name = f"{op_name}_{dtype}_vector"
                                external_kernels[func_name].append((op_name, dtype, shape))
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
                                external_kernels[func_name].append(("matmul", dtype, out_dtype, M, N, K))
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

    return external_kernels

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

def map_global_io(inputs, outputs) -> Tuple[Dict, int, int]:
    """
    Current constrians:
        - use 4 mem-shim tile pairs for io
        - each port is assigned to one dtensor tile
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

class ContextTransformer:
    def __init__(self, target_ctx, module):
        self.new_ctx = target_ctx
        self.aie_module = module
        self.variable_map = {}
        pass

    def get_ctx(self):
        return self.new_ctx
    
    def copy_op(self, allo_op:allo_ir.ir.Operation):
        new_attrs = None
        if allo_op.attributes:
            new_attrs = {}
            for i in range(len(allo_op.attributes)):
                named_attr = allo_op.attributes[i]
                new_attrs[named_attr.name] = named_attr.attr
        new_operands = None
        if allo_op.operands:
            new_operands = []
            for i in range(len(allo_op.operands)):
                new_operands.append(self.variable_map[allo_op.operands[i]])
        result_type = None
        if allo_op.results:
            result_type:List[aie_ir.Type] = []
            for i in range(len(allo_op.results)):
                print(allo_op.results[i].type)
                result_type.append(allo_op.results[i].type)
                print(allo_op.results[i].type, type(allo_op.results[i].type))
        print(allo_op)
        print(allo_op.name)
        print(new_operands)
        print(new_attrs)
        print(result_type)
        loc = aie_ir.Location.unknown()
        # 结果类型是 index
        idx_type = aie_ir.IndexType.get()
        # 创建属性：DenseElementsAttr 或 IntegerAttr 取决于 op 需要
        value_attr = aie_ir.IntegerAttr.get(idx_type, 0)
        # arith.constant 需要一个 attribute "value"
        attributes = {"value": value_attr}
        # 没有 operands
        operands = []
        # 一个结果，类型是 index
        results = [idx_type]
        print(results)
        print(attributes)
        op = aie_ir.Operation.create(
            name="arith.constant",
            results=results,
            operands=operands,
            attributes=attributes,
            loc=loc,
        )
        new_op = aie_ir.Operation.create(
            name="arith.constant",
            operands=[],
            attributes=attributes,
            results=results,
            loc = aie_ir.Location.unknown()
        )
        
        return new_op
    
    def transform(self, allo_module:allo_ir.ir.Operation):
        def dfs_transform(op, indent=0):
            for region in op.regions:
                for block in region.blocks:
                    for child_op in block.operations:
                        dfs_transform(child_op, indent + 1)
                        return
            new_op = self.copy_op(op)

            for old_result, new_result in zip(op.results, new_op.results):
                self.variable_map[old_result] = new_result
            print(new_op)
            
        dfs_transform(allo_module)

    def traverse(self, allo_module:allo_ir.ir.Operation):
        def dfs_traverse(op, indent=0):
            op_name = str(op.name)
            if '.' in op_name:
                dialect = op_name.split('.')[0]
            else:
                dialect = "(x)"
            
            print('  ' * indent + f"Operation: {op_name},\tDialect: {dialect}")
            for i in range(len(op.results)):
                print('  ' * indent + f"o:\t{op.results[i]}")

            for i in range(len(op.operands)):
                print('  ' * indent + f"i:\t{op.operands[i]}")

            for region in op.regions:
                for block in region.blocks:
                    for child_op in block.operations:
                        dfs_traverse(child_op, indent + 1)
        dfs_traverse(allo_module)

def build_compute_core(
    core_function: func_d.FuncOp
) -> aie_d.CoreOp:
    
    op_str = """
    module {
        aie.device(npu1_1col) {
            %tile_0_2 = aie.tile(0, 2)
            %core_0_2 = aie.core(%tile_0_2) {
                aie.end
            }
        }
    }
    """
    sample = aie_ir.Operation.parse(op_str, context=aie_ir.Context())
    print(sample)
    # TODO
    pass
   

def aie_codegen(
    device_type:str,
    core_func_groups:Dict[str, List[func_d.FuncOp]],
    external_funcs:List[func_d.FuncOp],
    inputs, outputs,
) -> aie_ir.Module:
    
    io_mapping, mem_tile_num, shim_tile_num = map_global_io(inputs, outputs)

    wrapper_code = f"""
        module {{
            aie.device({device_type}) {{
    """

    # fixme: maybe better to resolve this using IR constructor
    for func in external_funcs:
        wrapper_code += format_str(str(func), indent=4)
        
    wrapper_code += """
            }
        }
    """
    with aie_ir.Context() as ctx, aie_ir.Location.unknown():
        # module wrapper
        module = aie_ir.Module.parse(wrapper_code, ctx)
        # find device op: aie.device(device_type)
        device_op = None
        for op in module.body.operations:
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
        tile_map:Dict[str,aie_d.TileOp] = {}
        with aie_ir.InsertionPoint(end_op):
            # shim tile
            for shim_id in range(shim_tile_num):
                tile_map[f"shim_{shim_id}"] = aie_d.TileOp(
                    col=shim_id, row=0,
                    # allocation_scheme="shim",
                )
            # mem tiles
            for mem_id in range(mem_tile_num):
                tile_map[f"mem_{mem_id}"] = aie_d.TileOp(
                    col=mem_id, row=1,
                    # allocation_scheme="mem", 
                )
        print(module)

        transformer = ContextTransformer(ctx, module)
        # TODO
        # # build mlir code for each core
        # for func_name, funcs in core_func_groups.items():
        #     transformer.transform(funcs)

    return module

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
        external_kernels = inject_external_kernels(self.allo_module)
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
        self.aie_module = aie_codegen(
            device_type, core_func_groups, external_funcs,
            inputs, outputs,
        )
        pass
        # TODO

    def __call__(self, *args):
        pass
        # TODO