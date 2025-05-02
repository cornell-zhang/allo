import os
from typing import Dict, List, Tuple, Set

import allo._mlir._mlir_libs._mlir as allo_ir

from ..._mlir.dialects import func as allo_func_d

import aie.ir as aie_ir

from ...passes import analyze_read_write_patterns
from ..._mlir.passmanager import PassManager as mlir_pass_manager
from ...memory import DTensor
from .mlir_codegen import CodeGenerator
from .utils import inject_external_kernels, classify_aie_functions

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
        self.core_func_args:Dict[str,Tuple[DTensor,bool]] = {} # core func name -> a list of (dtensors, is_in)

        self.aie_module:aie_ir.Module = None
        print(module)
    
    def collect_io(
        self,
        func_groups:Dict[str, List[allo_func_d.FuncOp]],
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
                self.core_func_args[func_name_w_id] = []
                # [NOTE]: function name implies some kind of mapping from io tensor to 'core's
                func_id = tuple(map(int, func_name_w_id.split(func_name + "_")[-1].split("_")))
                # fixme: `analyze_read_write_patterns` considers parameters that are both read and written as outputs
                in_idx, out_idx = analyze_read_write_patterns(func)
                for io_lst, io_idx, io in ((inputs, in_idx, "in"), (outputs, out_idx, "out")):
                    io_lst[func_name][func_id] = []
                    for idx in io_idx:
                        dtensor = self.func_args[func_name_w_id][idx]
                        self.core_func_args[func_name_w_id].append((dtensor, io == "in"))
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
        top_func, core_func_groups, external_funcs = classify_aie_functions(self.allo_module)
        # TODO: maybe use other ways to capture the relationship between DTensor, function group
        inputs, outputs = self.collect_io(core_func_groups)

        code_generator = CodeGenerator(device_type)
        self.aie_module = code_generator.aie_codegen(
            core_func_groups, external_funcs,
            inputs, outputs,
            use_external_kernels,
            self.stream_info,
            self.core_func_args
        )
        # TODO

    def __call__(self, *args):
        pass
        # TODO