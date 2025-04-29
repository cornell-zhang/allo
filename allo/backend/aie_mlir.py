import os
from typing import Dict, List, Tuple

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

# =======================
import aie.dialects.aie as aie_d
import aie.dialects.aiex as aiex_d
import aie.dialects.aievec as aievec_d

import aie.ir as aie_ir
# =======================

def process_aie_functions(
    module: allo_ir.ir.Module
) -> Tuple[func_d.FuncOp, List[func_d.FuncOp], Dict]:
    # top function
    top_func:func_d.FuncOp = None
    # core functions
    core_funcs:List[func_d.FuncOp] = []
    # external kernel functions
    external_kernels = {}
    injected_kernels = set()
    with module.context, allo_ir.ir.Location.unknown():
        for func in module.body.operations:
            if isinstance(func, func_d.FuncOp) and (
                "sym_visibility" not in func.attributes
                or func.attributes["sym_visibility"].value != "private"
            ):
                if func.attributes["sym_name"].value == "top":
                    top_func = func
                else:
                    core_funcs.append(func)
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

    return top_func, core_funcs, external_kernels

class ContextTransformer:
    def __init__(self):
        self.new_ctx = aie_ir.Context()
        self.variable_map = {}
        pass

    def transform(self):
        pass

    def traverse(self):
        pass
    
def clone_operation(op: allo_ir.ir.Operation, mapping, ip):
    if len(op.regions)==0:
        return aie_ir.Operation.parse(op.to_asm(), context=aie_ir.Context())
    
    # Clone the operation into insertion point ip
    new_operands = [mapping[o] for o in op.operands]
    new_op = aie_ir.Operation.create(
        op.name,
        operands=new_operands,
        attributes=dict(op.attributes),
        results=op.results.type if len(op.results) > 0 else None,
        regions=[]
    )
    ip.insert(new_op)
    
    # Map results
    for old, new in zip(op.results, new_op.results):
        mapping[old] = new

    # Clone regions recursively
    for old_region, new_region in zip(op.regions, new_op.regions):
        for old_block in old_region.blocks:
            new_block = aie_ir.Block.create_at_start(new_region, old_block.argument_types)
            for old_op in old_block.operations:
                clone_operation(old_op, mapping, InsertionPoint.at_block_end(new_block))
    
    return new_op

def build_compute_core(
    core_function: func_d.FuncOp
) -> aie_d.CoreOp:
    

    # compute_core = aie_d.CoreOp(
    #     result = 
    #     tile =
    # )
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
    # return compute_core

def aie_codegen(
    device_type:str,
    core_funcs:List[func_d.FuncOp],
) -> aie_ir.Module:
    wrapper_code = f"""
        module {{
            aie.device({device_type}) {{
            }}
        }}
    """
    with aie_ir.Context() as ctx, aie_ir.Location.unknown():
        # module wrapper
        module = aie_ir.Module.parse(wrapper_code, ctx)
        print(module)
        # TODO
        # build mlir code for each core
        for core_func in core_funcs:
            compute_core_op = build_compute_core(core_func)
            print(compute_core_op)

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

        self.aie_module:aie_ir.Module = None
        print(module)
    
    def build(self, device_type = "npu1_4col"):
        os.makedirs(os.path.join(self.project_dir, "build"), exist_ok=True)
        # record original allo mlir
        with open(
            os.path.join(self.project_dir, "original.mlir"), "w", encoding="utf-8"
        ) as f:
            f.write(str(self.allo_module))
        # - extract external kernels, classify functions
        top_func, core_funcs, external_kernels = process_aie_functions(self.allo_module)
        # - lower tensor to memref with registered pass
        passes = ["func.func(convert-linalg-to-affine-loops),lower-affine",]
        pipeline = f'builtin.module({",".join(passes)})'
        with self.allo_module.context:
            mlir_pass_manager.parse(pipeline).run(self.allo_module.operation)
        print_module(self.allo_module)
        print()
        print(self.allo_module)
        self.aie_module = aie_codegen(
            device_type, core_funcs
        )
        pass
        # TODO

    def __call__(self, *args):
        pass
        # TODO