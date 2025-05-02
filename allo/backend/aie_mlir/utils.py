import re

import aie.ir as aie_ir
import allo._mlir._mlir_libs._mlir as allo_ir
from typing import Dict, List, Tuple
from ..._mlir.dialects import func as allo_func_d

from ..._mlir.ir import (
    InsertionPoint,
    FlatSymbolRefAttr,
    StringAttr,
)

def inject_external_kernels(module: allo_ir.ir.Module) -> Dict[str,bool]:
    '''
    Inject external kernels for compute cores.

    Return:
        - use_external_kernels: func_name -> use external
    '''
    use_external_kernels = {}
    injected_kernels = set()
    with module.context, allo_ir.ir.Location.unknown():
        for func in module.body.operations:
            if isinstance(func, allo_func_d.FuncOp) and (
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
                                kernel_name = f"{op_name}_{dtype}_vector"
                                use_external_kernels[func_name] = True
                            # matmul
                            elif op.operation.name == "linalg.matmul":
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
                            func_type = allo_func_d.FunctionType.get(
                                [op.inputs[0].type, op.inputs[1].type, op.outputs[0].type], [],
                            )
                            # replace operation
                            allo_func_d.CallOp(
                                [], FlatSymbolRefAttr.get(kernel_name),
                                [op.inputs[0], op.inputs[1], op.outputs[0]],
                                ip=InsertionPoint(op),
                            )
                            op.erase()
                            # register external kernel
                            if kernel_name in injected_kernels:
                                continue
                            injected_kernels.add(kernel_name)
                            kernel = allo_func_d.FuncOp(
                                kernel_name, func_type, ip=InsertionPoint(func),
                            )
                            kernel.attributes["sym_visibility"] = StringAttr.get("private")
    return use_external_kernels

def classify_aie_functions(
    module: allo_ir.ir.Module
) -> Tuple[allo_func_d.FuncOp, Dict[str, List[allo_func_d.FuncOp]], List[allo_func_d.FuncOp]]:
    # top function
    top_func:allo_func_d.FuncOp = None
    # core functions
    core_func_groups:Dict[str, List[allo_func_d.FuncOp]] = {}
    # external functions
    external_funcs:List[allo_func_d.FuncOp] = []
    with module.context, allo_ir.ir.Location.unknown():
        for func in module.body.operations:
            if isinstance(func, allo_func_d.FuncOp):
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

def get_element_type(dtype_str: str) -> aie_ir.Type:
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
 
def dfs_print(op, indent=0):
    op_name = str(op.name)
    if '.' in op_name:
        dialect = op_name.split('.')[0]
    else:
        dialect = "(x)"
    
    print('  ' * indent + f"Operation: {op_name},\tDialect: {dialect}")

    for region in op.regions:
        for block in region.blocks:
            for child_op in block.operations:
                dfs_print(child_op, indent + 1)

def print_module(module):
    dfs_print(module.operation)